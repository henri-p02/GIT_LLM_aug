from collections.abc import Callable
from typing import Iterable, Literal, TypedDict, Union
import torch
import numpy as np
import wandb
import time
from datetime import timedelta


class OptimizerConfig(TypedDict):
    optim: Literal["Adam", "AdamW", "SGD"]
    base_lr: float
    args: dict


class TrainConfig(TypedDict):
    num_steps: int
    batches_per_step: int
    log_interval: int
    eval_interval: int
    optimizer: OptimizerConfig
    autocast: bool
    lr_scheduler: str
    warmup_steps: int
    label_smoothing: float
    clip_grad: Union[float, None]


def get_optim(config: TrainConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if not "args" in config["optimizer"]:
        config["optimizer"]["args"] = {}

    if "weight_decay" in config["optimizer"]["args"]:
        # apply weight decay only to specific parameters
        param_map = {p_name: p for p_name, p in model.named_parameters()}
        decay, no_decay = model.get_wd_params()
        optim_groups = [
            {
                "params": [param_map[name] for name in decay],
                "weigth_decay": config["optimizer"]["args"]["weight_decay"],
            },
            {"params": [param_map[name] for name in no_decay], "weight_decay": 0.0},
        ]
        config["optimizer"]["args"].pop("weight_decay")
    else:
        optim_groups = model.parameters()
    if config["optimizer"]["optim"] == "Adam":
        return torch.optim.Adam(
            optim_groups, config["optimizer"]["base_lr"], **config["optimizer"]["args"]
        )
    elif config["optimizer"]["optim"] == "AdamW":
        return torch.optim.AdamW(
            optim_groups, config["optimizer"]["base_lr"], **config["optimizer"]["args"]
        )
    else:  # config['optimizer']['optim'] == "SGD":
        return torch.optim.SGD(
            optim_groups, config["optimizer"]["base_lr"], **config["optimizer"]["args"]
        )


def get_scheduler(
    config: TrainConfig, optim: torch.optim.Optimizer, **args
) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    if config["lr_scheduler"] == "like_transformer":
        warmup_steps = config["warmup_steps"]
        assert isinstance(
            warmup_steps, int
        ), "LR Scheduler expects 'warmup_steps' to be an int"
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
            optim, config["num_steps"] - warmup_steps, eta_min=1e-5
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optim, [linear, cos], milestones=[warmup_steps]
        )
    elif config["lr_scheduler"] == "constant":
        return None
    elif config["lr_scheduler"] == "1cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=config["optimizer"]["base_lr"],
            total_steps=config["num_steps"] + 1,
        )
    else:
        raise Exception("Unknown LR scheduler")


def train_loop(
    model: torch.nn.Module,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
    calc_train_loss: Callable[
        [torch.nn.Module, tuple[torch.Tensor, torch.Tensor], torch.device], torch.Tensor
    ],
    train_loader: torch.utils.data.DataLoader,
    eval_model: Callable[
        [torch.nn.Module, torch.device], tuple[float, float, float, float]
    ],
    device: torch.device,
    cancel_at: Union[int, None] = None,
    early_stop=None,
):
    use_wandb = True if wandb.run is not None else False

    loss_curve = []
    accuracy_curve = []

    model.to(device)
    
    if torch.cuda.get_device_capability(device)[0] >= 7:
        model.compile()

    num_epochs = (
        config["num_steps"] // (len(train_loader) // config["batches_per_step"])
    ) + 1

    print("=========================================")
    print("Starting Traning")
    print(f"  -Using device: {device}")
    print(f"  -Number of parameters: {np.sum([p.numel() for p in model.parameters()])}")
    print(f'  -Number of steps: {config["num_steps"]}')
    print(f"  -Number of epochs: {num_epochs}")
    print(f"  -Batch size: {train_loader.batch_size}")
    print("=========================================")
    print()

    if config["autocast"]:
        scaler = torch.amp.grad_scaler.GradScaler(device=device.type)

    step = 0
    loss_history = []
    optimizer.zero_grad()
    start_time = time.time()
    for epoch in range(num_epochs):
        for batch_index, batch in enumerate(train_loader):
            if cancel_at is not None and time.time() >= cancel_at:
                print("### TIME IS UP! ###")
                return

            model.train()

            if config["autocast"]:
                with torch.autocast(device_type=device.type):
                    loss = calc_train_loss(model, batch, device)
                scaler.scale(loss / config["batches_per_step"]).backward()  # type: ignore
            else:
                loss = calc_train_loss(model, batch, device)
                (loss / config["batches_per_step"]).backward()

            loss_history.append(loss.detach().clone())

            if (batch_index + 1) % config["batches_per_step"] == 0:
                # Perform step
                step += 1

                if config["autocast"]:
                    if config["clip_grad"] is not None:
                        scaler.unscale_(optimizer)  # type: ignore
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["clip_grad"]
                        )
                    scaler.step(optimizer)  # type: ignore
                    scaler.update()  # type: ignore
                else:
                    if config["clip_grad"] is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config["clip_grad"]
                        )
                    optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if (step + 1) % config["log_interval"] == 0:
                    # Track train loss every few steps
                    average_loss = torch.stack(loss_history).mean().item()
                    loss_curve.append(average_loss)
                    loss_history = []

                    if use_wandb:
                        if lr_scheduler is not None:
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = config["optimizer"]["base_lr"]

                        wandb.log(
                            {"loss": average_loss, "learning_rate": current_lr},
                            step=step + 1,
                        )
                    print(
                        f'Epoch: {epoch + 1:3d}/{num_epochs:3d}, Step {step + 1:5d}/{config["num_steps"]}, Loss: {average_loss:.4f}'
                    )

                # After eval_every steps, calc accuracy
                if step % config["eval_interval"] == 0:
                    model.eval()
                    train_accuracy, train_entropy, test_accuracy, test_entropy = (
                        eval_model(model, device)
                    )
                    accuracy_curve.append(test_accuracy)
                    if use_wandb:
                        wandb.log(
                            {
                                "accuracy": test_accuracy,
                                "train_accuracy": train_accuracy,
                                "entropy": test_entropy,
                                "train_entropy": train_entropy,
                            },
                            step=step + 1,
                        )
                    print("=========================================")
                    print(
                        f'Epoch: {epoch + 1:3d}/{num_epochs:3d}, Step: {step + 1:5d}/{config["num_steps"]}, Accuracy: {test_accuracy * 100:.4f} %'
                    )
                    print("=========================================")

                if early_stop is not None:
                    if step >= early_stop:
                        return

        epoch_time = time.time() - start_time
        if use_wandb:
            wandb.log({"epoch_time": epoch_time}, step=step + 1)
        if epoch % 10 == 0:
            eta = epoch_time * (num_epochs - epoch - 1)
            print(
                f"+++ Epoch {epoch + 1} took: {epoch_time}s, ETA: {timedelta(seconds=eta)}"
            )
        start_time = time.time()

    return loss_curve, accuracy_curve


def run_training(
    model: torch.nn.Module,
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None],
    calc_train_loss: Callable[
        [torch.nn.Module, tuple[torch.Tensor, torch.Tensor], torch.device], torch.Tensor
    ],
    train_loader: torch.utils.data.DataLoader,
    eval_model: Callable[
        [torch.nn.Module, torch.device], tuple[float, float, float, float]
    ],
    device: torch.device,
    cancel_at: Union[int, None] = None,
    early_stop=None,
):
    try:
        train_loop(
            model=model,
            config=config,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            calc_train_loss=calc_train_loss,
            train_loader=train_loader,
            eval_model=eval_model,
            device=device,
            cancel_at=cancel_at,
            early_stop=early_stop,
        )
    except Exception as error:
        if wandb.run is not None:
            wandb.finish(1)
        raise error
    else:
        if wandb.run is not None:
            wandb.finish()
