from collections.abc import Callable
from typing_extensions import Iterable, Literal, TypedDict, Union
import torch
import numpy as np
import wandb
import time
import logging
import os
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

logger = logging.getLogger(__name__)


class OptimizerConfig(TypedDict):
    optim: Literal["Adam", "AdamW", "SGD"]
    base_lr: float
    args: dict


class TrainConfig(TypedDict):
    num_steps: int
    resulting_batch_size: int
    batches_per_step: int
    log_interval: int
    eval_interval: int
    optimizer: OptimizerConfig
    autocast: bool
    lr_scheduler: str
    warmup_steps: int
    label_smoothing: float
    clip_grad: Union[float, None]
    early_stopping: Union[int, None]


def uniquify_path(path):
    """If path already exists, add a counter, to not override the file"""
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "-" + str(counter) + extension
        counter += 1

    return path


def checkpoint_model(
    model, optimizer, step, best=False, override=False, wandb_run=None
):
    """Save a checkpoint of the given model and optimizer to resume training later."""
    if wandb_run is not None:
        filename = f"coco-{wandb_run.name}.pt"
    else:
        filename = "coco-caption.pt"

    filename = "best-" + filename if best else filename
    filename = "models/" + filename

    filename = filename if override else uniquify_path(filename)

    model_to_save = model.module if isinstance(model, DDP) else model

    torch.save(
        {
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        filename,
    )


def get_optim(config: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build optimizer for the model"""
    if not "args" in config["optimizer"]:
        config["optimizer"]["args"] = {}

    groups = model.get_param_groups()

    decay, no_decay, image_encoder = groups

    if "weight_decay" in config["optimizer"]["args"]:
        optim_groups = [
            {
                "params": [p for p in decay],
                "weigth_decay": config["optimizer"]["args"]["weight_decay"],
            },
            {"params": [p for p in no_decay], "weight_decay": 0.0},
        ]
    else:
        all_params = decay + no_decay
        optim_groups = [{"params": [p for p in all_params], "weight_decay": 0.0}]

    optim_groups.append(
        {
            "params": [p for p in image_encoder],
            "weight_decay": 0.0,
            "lr": config["optimizer"]["base_lr"] * 0.2,
        }
    )

    if config["optimizer"]["optim"][:5] == "AdamW":
        return torch.optim.AdamW(
            optim_groups, config["optimizer"]["base_lr"], **config["optimizer"]["args"]
        )
    elif config["optimizer"]["optim"][:4] == "Adam":
        return torch.optim.Adam(
            optim_groups, config["optimizer"]["base_lr"], **config["optimizer"]["args"]
        )
    else:  # config['optimizer']['optim'] == "SGD":
        return torch.optim.SGD(
            optim_groups, config["optimizer"]["base_lr"], **config["optimizer"]["args"]
        )


def get_scheduler(
    config: TrainConfig, optim: torch.optim.Optimizer, **args
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler for the given optimizer"""

    if config["lr_scheduler"] == "like_transformer":
        warmup_steps = config["warmup_steps"]
        lr_func = lambda step: args["d_model"] ** (-0.5) * min(
            (step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)
        )
        return torch.optim.lr_scheduler.LambdaLR(optim, lr_func)
    elif config["lr_scheduler"] == "warmup_cosine":
        warmup_steps = config["warmup_steps"]
        linear = torch.optim.lr_scheduler.LinearLR(
            optim, 1 / 3, 1, total_iters=warmup_steps
        )
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, config["num_steps"] - warmup_steps, eta_min=0
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optim, [linear, cos], milestones=[warmup_steps]
        )
    elif config["lr_scheduler"] == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 1)
    elif config["lr_scheduler"] == "1cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=config["optimizer"]["base_lr"],
            total_steps=config["num_steps"] + 1,
        )
    else:
        raise Exception("Unknown LR scheduler")


def train_loop(
    rank: int,
    model: torch.nn.Module,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
    calc_train_loss: Callable[
        [torch.nn.Module, list[torch.Tensor], torch.device], torch.Tensor
    ],
    train_loader: Iterable,
    eval_model: Callable[[torch.nn.Module, torch.device], tuple[torch.Tensor, float]],
    device: torch.device,
    cancel_at: Union[int, None] = None,
    eval_example=Union[Callable[[torch.nn.Module], dict], None],
    wandb_run=None,
    **kwargs,
):
    use_wandb = wandb_run is not None
    is_ddp = isinstance(model, DDP)

    if use_wandb:
        if is_ddp:
            wandb_run.watch(model.module, log="all")
        else:
            wandb_run.watch(model, log="all")

    best_cider = 0

    logger.debug(f"Moving model to {device}...")
    model.to(device)
    if torch.cuda.get_device_capability(device)[0] >= 7:
        logger.debug("Compiling model...")
        model.compile()

    print("=========================================")
    print("Starting Traning")
    print(f"  -Using device: {device}")
    print(f"  -Number of parameters: {np.sum([p.numel() for p in model.parameters()])}")
    print(f'  -Number of steps: {config["num_steps"]}')
    print("=========================================")
    print()

    if config["autocast"]:
        scaler = torch.amp.grad_scaler.GradScaler(device=device.type)

    step = 0
    epoch = 0
    loss_history = []
    optimizer.zero_grad()
    start_time = time.time()

    epoch_limit = (
        100_000
        if config["early_stopping"] is None and config["early_stopping"] != 0
        else config["early_stopping"]
    )

    while epoch < epoch_limit:
        epoch += 1

        for batch_index, batch in enumerate(train_loader):
            if cancel_at is not None and time.time() >= cancel_at:
                print("### TIME IS UP! ###")
                return

            model.train()

            if config["autocast"]:
                with torch.autocast(device_type=device.type):
                    loss = calc_train_loss(model, batch, device)
                scaler.scale(loss / config["batches_per_step"]).backward()  # type:ignore
            else:
                loss = calc_train_loss(model, batch, device)
                (loss / config["batches_per_step"]).backward()

            loss_history.append(loss.detach().clone())

            # Perform step after batches_per_step backwards
            if (batch_index + 1) % config["batches_per_step"] == 0:
                step += 1

                if config["autocast"]:
                    if config["clip_grad"] is not None:
                        scaler.unscale_(optimizer)  # type:ignore
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["clip_grad"]
                        )
                    scaler.step(optimizer)  # type:ignore
                    scaler.update()  # type:ignore
                else:
                    if config["clip_grad"] is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["clip_grad"]
                        )
                    optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Track train loss every few steps
                if step % config["log_interval"] == 0:
                    average_loss = torch.stack(loss_history).mean()
                    loss_history = []
                    if is_ddp:
                        dist.all_reduce(average_loss, dist.ReduceOp.AVG)

                    average_loss = average_loss.item()
                    if rank == 0:
                        if lr_scheduler is not None:
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = config["optimizer"]["base_lr"]

                        if use_wandb:
                            wandb_run.log(
                                {
                                    "loss": average_loss,
                                    "learning_rate": current_lr,
                                    "num_samples": step
                                    *config["resulting_batch_size"],
                                },
                                step=step,
                            )
                        logger.debug(
                            {"loss": average_loss, "learning_rate": current_lr}
                        )
                        print(
                            f'Epoch: {epoch:3d}/{epoch_limit or -1:3d}, Step {step:5d}/{config["num_steps"]}, Loss: {average_loss:.4f}'
                        )

                # After eval_interval steps, calc accuracy
                if step % config["eval_interval"] == 0:
                    if rank == 0:
                        model.eval()
                        cider_scores, val_loss = eval_model(model, device)
                        cider_mean = cider_scores.mean()
                        cider_std = cider_scores.std()
                        cider_hist = wandb.Histogram(cider_scores) # type:ignore
                        if use_wandb:
                            wandb_run.log(
                                {
                                    "validation_loss": val_loss,
                                    "cider_mean": cider_mean,
                                    "cider_std": cider_std,
                                    "cider_values": cider_hist,
                                    "num_samples": step
                                    * config["resulting_batch_size"],
                                },
                                step=step,
                            )
                        logger.debug(
                            {
                                "validation_loss": val_loss,
                                "cider_mean": cider_mean,
                                "cider_std": cider_std,
                                "cider_values": cider_hist,
                            }
                        )
                        if cider_mean < best_cider:
                            checkpoint_model(
                                model,
                                optimizer,
                                step,
                                best=True,
                                override=True,
                                wandb_run=wandb_run,
                            )
                            best_cider = cider_mean
                        print("=========================================")
                        print(
                            f'Epoch: {epoch:3d}/{epoch_limit or -1:3d}, Step: {step:5d}/{config["num_steps"]}, CIDEr-D score: {cider_mean * 100:.4f} %'
                        )
                        print("=========================================")
                        if eval_example is not None:
                            out: dict = eval_example(model) # type: ignore
                            out["num_samples"] = step * config["resulting_batch_size"]
                            logger.debug(out)
                            if use_wandb:
                                wandb_run.log(out, step=step)

                    # other processes wait for rank 0 to finish eval
                    if torch.distributed.is_initialized():
                        torch.distributed.barrier()

                # End training if num_steps were performed
                if step >= config["num_steps"]:
                    return

        epoch_time = time.time() - start_time
        if is_ddp:
            epoch_time = torch.tensor(epoch_time, device=device)
            dist.all_reduce(epoch_time, dist.ReduceOp.AVG)
            epoch_time = epoch_time.item()

        if rank == 0:
            if use_wandb:
                wandb_run.log(
                    {
                        "epoch_time": epoch_time,
                        "num_samples": step * config["resulting_batch_size"],
                    },
                    step=step,
                )
            logger.debug({"epoch_time": epoch_time})
            eta = epoch_time * (config["num_steps"] - step - 1) / (step + 1) * epoch
            print(
                f"+++ Epoch {epoch} took: {epoch_time}s, ETA: {timedelta(seconds=eta)}"
            )

        start_time = time.time()


def train(
    rank: int,
    model: torch.nn.Module,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
    calc_train_loss: Callable[
        [torch.nn.Module, list[torch.Tensor], torch.device], torch.Tensor
    ],
    train_loader: Iterable,
    eval_model: Callable[[torch.nn.Module, torch.device], tuple[torch.Tensor, float]],
    device: torch.device,
    cancel_at: Union[int, None] = None,
    eval_example: Union[Callable[[torch.nn.Module], dict], None] = None,
    wandb_run=None,
    **kwargs,
):
    try:
        train_loop(
            rank=rank,
            model=model,
            config=config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calc_train_loss=calc_train_loss,
            train_loader=train_loader,
            eval_model=eval_model,
            device=device,
            cancel_at=cancel_at,
            eval_example=eval_example, #type: ignore
            wandb_run=wandb_run,
            **kwargs,
        )
    except Exception as error:
        if rank == 0 and wandb_run is not None:
            wandb_run.finish(1)
        raise error
    else:
        if rank == 0 and wandb_run is not None:
            wandb_run.finish()
    finally:
        if hasattr(train_loader, "_iterator"):
            logger.debug("Cleaning up train iterator")
            del train_loader._iterator #type:ignore
        if rank == 0:
            checkpoint_model(
                model, optimizer, None, best=False, override=False, wandb_run=wandb_run
            )
