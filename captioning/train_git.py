from typing import TypedDict
from utils_datasets import (
    DatasetConfig,
    get_caption_dataset,
    reduce_dl_size,
    build_dataloader,
    get_tokenizer_from_vocab,
    cut_caption,
)
from own_git import GITConfig, get_caption_model, GITCaptioning
from utils_training import TrainConfig, get_optim, get_scheduler, train
import torch
import torch.nn as nn
import argparse
import logging
import wandb
from functools import partial
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import warnings
import sys

sys.path.append("./metrics")
from cider import CIDER

warnings.filterwarnings("ignore", category=FutureWarning)

T = torch.Tensor
logging.basicConfig(level=logging.INFO)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class Config(TypedDict):
    model: GITConfig
    dataset: DatasetConfig
    train: TrainConfig


config: Config = {
    "model": {
        "cross_attention": False,
        "image_encoder": "CLIP/ViT-B-16",
        "vocab_size": 28999,
        "max_seq_len": 30,
        "share_embedding": True,
        "d_model": 768,
        "d_ffn": 1396,
        "num_layers": 5,
        "num_heads": 16,
        "dropout": 0,
        "torch_attn": True,
        "init_embedding": None,
        "gpt_embedding": False,
        "pos_encoding": "learned",
    },
    "dataset": {
        "dataset": "COCO-karpathy",
        "eval_dataset": "COCO-karpathy",
        "batch_size": 256,
        "eval_batch_size": 1024,
        "num_workers": 24,
        "prefetch_factor": 4,
        "augmentation": "flip-perspective",
        "grouped": False,
    },
    "train": {
        "resulting_batch_size": 4096,
        "batches_per_step": 0,  # calculated later in code
        "num_steps": 100_000,
        "warmup_steps": 500,
        "early_stopping": 100,
        "optimizer": {
            "optim": "AdamW",
            "base_lr": 4e-5,
            "args": {"weight_decay": 0.00412, "betas": (0.9, 0.99)},
        },
        "lr_scheduler": "warmup_cosine",
        "autocast": True,
        "clip_grad": None,
        "log_interval": 100,
        "eval_interval": 200,
        "label_smoothing": 0.02,
    },
}


def calc_train_loss_git(
    model: nn.Module, batch: list[T], device: torch.device, criterion, tokenizer
) -> T:
    """Calculate loss value for one batch"""

    model.train()
    im, caption = batch[0].to(device, non_blocking=True), batch[1].to(
        device, non_blocking=True
    )

    logits: T = model(im, caption)

    tgt = caption[:, 1:]
    logits = logits[:, :-1]

    eos_pos = tgt == tokenizer.eos_token_id
    lens = eos_pos.max(dim=-1)[1] + 1
    pad_mask = torch.arange(tgt.size(-1), device=device) < lens[:, None]

    tgt = tgt[pad_mask]
    logits = logits[pad_mask]

    loss = criterion(logits, tgt)
    return loss


def calc_val_loss(
    model: nn.Module, device: torch.device, eval_loader, criterion, tokenizer
):
    """Calc loss on validation dataset"""

    model.eval()
    sum_loss = torch.empty((0,), device=device)
    num_samples = torch.tensor(0, device=device)
    for img, cap in eval_loader:
        img, cap = img.to(device, non_blocking=True), cap.to(device, non_blocking=True)
        num_samples += img.size(0)
        with torch.no_grad():
            logits = model(img, cap)

            tgt = cap[:, 1:]
            logits = logits[:, :-1]

            eos_pos = tgt == tokenizer.eos_token_id
            lens = eos_pos.max(dim=-1)[1] + 1
            pad_mask = torch.arange(tgt.size(-1), device=device) < lens[:, None]

            tgt = tgt[pad_mask]
            logits = logits[pad_mask]

            loss = criterion(logits, tgt)
            sum_loss = torch.cat((sum_loss, loss.detach().clone().unsqueeze(0)))

    val_loss = sum_loss.mean().item()
    logger.debug(f"Consumed {num_samples.item()} samples for validation loss")
    return val_loss


def eval_model_cider(
    model: nn.Module,
    device: torch.device,
    eval_loader,
    batched_eval_loader,
    criterion,
    tokenizer,
):
    """Evaluate model performance (loss and CIDEr-D)"""
    val_loss = calc_val_loss(model, device, eval_loader, criterion, tokenizer)

    scorer = CIDER(4)
    model.eval()

    num_samples = torch.tensor(0, device=device)
    for img, refs in batched_eval_loader:
        img = img.to(device, non_blocking=True)
        num_samples += img.size(0)
        with torch.no_grad():
            real_model = model.module if isinstance(model, DDP) else model
            candidates: T = real_model.infer(img, selector="max")
        candidates = cut_caption(candidates, tokenizer).detach().clone()

        for cand, refs in zip(candidates, refs):
            cand = cand[cand != tokenizer.pad_token_id]
            refs = [(ref[ref != tokenizer.pad_token_id]).tolist() for ref in refs]

            scorer.add_sample(cand.tolist(), refs)

    cider_scores = scorer.calc_all_samples(silent=True)
    logger.debug(f"Consumed {num_samples.item()} samples for cider score")
    return cider_scores, val_loss


def eval_example(model: torch.nn.Module, fixed_ims, tokenizer, device):
    """Perform inference on fixed images to compare performance"""
    model.eval()
    infer_model = model.module if isinstance(model, DDP) else model

    with torch.no_grad():
        with torch.autocast(device_type=device.type):
            caps = infer_model.infer(fixed_ims)

    # normalize images for visualization
    ims = fixed_ims.clone().permute(0, 2, 3, 1)
    ims = ims - ims.min()
    ims = ims / ims.max()
    logger.debug(ims.shape)
    caps = tokenizer.batch_decode(caps)
    img_obj = [
        wandb.Image(ims[i].cpu().numpy(), caption=caps[i]) for i in range(ims.size(0))
    ]
    return {"examples": img_obj}


def ddp_setup(rank: int, world_size: int, gpu_ids: list[int]):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(gpu_ids[rank])
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def run(rank, args: argparse.Namespace, config: Config, wandb_run=None):
    world_size = len(args.cuda)
    is_ddp = world_size > 1

    logger = logging.getLogger(f"{__name__}-rank{rank}")
    if args.test:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not is_ddp:
        assert rank == 0, "If non ddp, rank must be 0"
        device = torch.device("cuda:" + str(args.cuda[0]))
    else:
        ddp_setup(rank, world_size, args.cuda)
        logger.debug("Created DDP Context!")
        device = torch.device("cuda:" + str(args.cuda[rank]))

    if args.finetune is not None:
        logger.debug("Getting original model config")
        orig_run = wandb.Api().run(args.finetune)
        config["model"] = orig_run.config["model"]

    # limit batch_size
    if config["dataset"]["batch_size"] // world_size > 256:
        config["dataset"]["batch_size"] = 256 * world_size
        logger.info(
            f"Reducing batch size for one forward pass to {config['dataset']['batch_size']} for world size {world_size}"
        )

    # adjust batches_per_step to match resulting_batch_size
    if (
        config["train"]["resulting_batch_size"] is not None
        and config["train"]["resulting_batch_size"] > 0
    ):
        config["train"]["batches_per_step"] = (
            config["train"]["resulting_batch_size"] // config["dataset"]["batch_size"]
        )
        logger.info(
            f"Using {config['train']['batches_per_step']} batches per step to get effective batch size of {config['train']['resulting_batch_size']}"
        )

    tokenizer = get_tokenizer_from_vocab(config["model"]["vocab_size"])

    train_set, _ = get_caption_dataset(
        config["dataset"], "/home/poeche/ws/ba/code/data/", tokenizer, world_size
    )
    _, val_set = get_caption_dataset(
        config["dataset"], "/home/poeche/ws/ba/code/data/", tokenizer, 1
    )  # validation only on rank 0

    batched_ds_conf: DatasetConfig = {
        "dataset": "COCO-karpathy",
        "batch_size": 256,
        "eval_batch_size": 512,
        "num_workers": config["dataset"]["num_workers"],
        "prefetch_factor": 4,
        "augmentation": "",
        "grouped": True,
    }
    _, batched_val_set = get_caption_dataset(
        batched_ds_conf, "/home/poeche/ws/ba/code/data/", tokenizer, 1
    )

    logger.debug("Building model...")
    model = get_caption_model(config["model"], tokenizer)
    logger.debug("Model built")

    logger.debug("Building optimizer and scheduler...")
    optim = get_optim(config["train"], model)
    lr_scheduler = get_scheduler(config["train"], optim)
    logger.debug("Optimizer and scheduler built")

    if args.checkpoint:
        logger.debug("Reading checkpoint data...")
        checkpoint_data = torch.load(
            args.checkpoint, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint_data["model"])
        model.to(device)
        logger.debug("Loaded model weights!")
        optim.load_state_dict(checkpoint_data["optimizer"])

        if args.resume:
            step_to_resume = checkpoint_data["step"]
            if step_to_resume is None:
                if args.resume_step is None:
                    raise ValueError("No step provided. Set --resume-step")
                else:
                    step_to_resume = args.resume_step
            for _ in range(step_to_resume):
                lr_scheduler.step()

    criterion = nn.CrossEntropyLoss(label_smoothing=config["train"]["label_smoothing"])

    train_loader, eval_loader = build_dataloader(config["dataset"], train_set, val_set)
    _, batched_eval_loader = build_dataloader(
        batched_ds_conf, torch.utils.data.Dataset(), batched_val_set
    )

    additional_parameters = {}

    if args.test:
        config["train"]["num_steps"] = 10
        config["train"]["log_interval"] = 1
        config["train"]["eval_interval"] = 5

        train_loader, eval_loader = reduce_dl_size(
            train_loader, eval_loader, config["train"]["batches_per_step"] * 2, 16
        )

    if rank == 0:
        fixed_ims = torch.load(
            "./example_data.pth", map_location="cpu"
        )["images"].to(device)
        example_func = partial(
            eval_example, fixed_ims=fixed_ims, device=device, tokenizer=tokenizer
        )
    else:
        example_func = None

    if is_ddp:
        logger.debug(f"Dataparallel: Copying model to device {args.cuda[rank]}")
        model = model.to(args.cuda[rank])
        model = DDP(model, device_ids=[args.cuda[rank]], find_unused_parameters=False)
        logger.debug(f"Model copied!")

    cancel_at = None

    try:

        train(
            rank=rank,
            model=model,
            config=config["train"],
            optimizer=optim,
            lr_scheduler=lr_scheduler,
            calc_train_loss=partial(
                calc_train_loss_git, criterion=criterion, tokenizer=tokenizer
            ),
            train_loader=train_loader,
            eval_model=partial(
                eval_model_cider,
                eval_loader=eval_loader,
                batched_eval_loader=batched_eval_loader,
                criterion=criterion,
                tokenizer=tokenizer,
            ), #type: ignore
            device=device,
            cancel_at=cancel_at,
            eval_example=example_func,
            wandb_run=wandb_run,
            **additional_parameters,
        )
    except Exception as error:
        logger.exception(error)
        raise
    finally:
        if is_ddp:
            destroy_process_group()


def is_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} not found")


def parse_arguments():
    parser = argparse.ArgumentParser("train a git on a dataset")
    parser.add_argument("--cuda", action="append", required=True, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--finetune", type=str)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--resume_step", type=int)
    parser.add_argument("--checkpoint", type=is_file)
    parser.add_argument("--config", type=is_file)

    args = parser.parse_args()

    if args.finetune is not None and args.checkpoint is None:
        raise argparse.ArgumentError(
            None, "If --finetune is set, --checkpoint needs to be set too."
        )
    if args.resume is not None and args.checkpoint is None:
        raise argparse.ArgumentError(
            None, "If --resume is set, --checkpoint needs to be set too."
        )
    if args.checkpoint is not None and args.finetune is None and args.resume is None:
        raise argparse.ArgumentError(
            None, "If --checkpoint is set, --finetune or --resume needs to be set too."
        )

    return args


if __name__ == "__main__":
    args = parse_arguments()

    if args.resume:
        logger.debug("Getting original config")
        orig_run = wandb.Api().run(args.resume)
        config = orig_run.config
        logger.debug(config)

    if args.test:
        logging.basicConfig(level=logging.DEBUG)
        wandb_run = None
    else:
        logging.basicConfig(level=logging.INFO)
        wandb_run = wandb.init(project="git-hp-search", config=config)  # type: ignore

    is_ddp = len(args.cuda) > 1

    if len(args.cuda) == 1:
        run(0, args, config, wandb_run)
    elif len(args.cuda) > 1:
        mp.spawn(run, args=(args, config, wandb_run), nprocs=len(args.cuda)) #type: ignore
    else:
        raise Exception("Minimum one GPU needs to be defined")