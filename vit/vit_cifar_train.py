# %%
import logging
from typing import Any, Dict, TypeVar, Callable, TypedDict
import own_vit
import training_utils
import dataset_utils
import torch
import wandb
import argparse
import time
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

# %%
model_base_config: own_vit.ViTConfig = {
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 6,
    "d_ffn": 512,
    "dropout": 0.2,
    "image_size": 32,
    "image_channels": 3,
    "patch_size": 4,
    "out_classes": 10,
    "conv_proj": True,
    "torch_att": True,
}

train_base_config: training_utils.TrainConfig = {
    "num_steps": 100_000,
    "warmup_steps": 3_900,
    "optimizer": {"optim": "AdamW", "base_lr": 5e-4, "args": {"weight_decay": 0.03}},
    "batches_per_step": 1,
    "eval_interval": 1000,
    "log_interval": 100,
    "autocast": True,
    "lr_scheduler": "warmup_cosine",
    "label_smoothing": 0.1,
    "clip_grad": None,
}

dataset_base_config: dataset_utils.DatasetConfig = {
    "dataset": "CIFAR10",
    "augmentation": "AutoCIFAR10-MixUp",
    "batch_size": 128,
    "num_workers": 8,
}


class Config(TypedDict):
    model: own_vit.ViTConfig
    train: training_utils.TrainConfig
    dataset: dataset_utils.DatasetConfig


base_config: Config = {
    "model": model_base_config,
    "train": train_base_config,
    "dataset": dataset_base_config,
}

# train mulitple variations sequentially, by defining dict of updates on the base_config
variations = [{}]

# %%
KeyType = TypeVar("KeyType")


def deep_update(
    mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def get_loss_fn(
    model_config: own_vit.ViTConfig,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataset_config: dataset_utils.DatasetConfig,
):
    """Returns a function to compute the loss of a model on a batch of training data

    Args:
        model_config (ViTConfig): Config object for the model to compute loss on
        criterion (Function): The criterion function, taking the predicted logits and target labels, returning a float loss value
        dataset_config (dataset_utils.DatasetConfig): Config object for the dataset. Used to apply CutMix and/or MixUp, which operate on batches rather than individual samples

    Returns:
        Function: A function, that can be called with the model, a batch of the training data and a device to calculate on
    """
    if dataset_config["augmentation"] is not None:
        if (
            "MixUp" in dataset_config["augmentation"]
            and "CutMix" in dataset_config["augmentation"]
        ):
            mix = v2.RandomChoice(
                [
                    v2.MixUp(num_classes=model_config["out_classes"]),
                    v2.CutMix(num_classes=model_config["out_classes"]),
                    v2.Identity(),
                ],
                p=[0.25, 0.25, 0.5],
            )
        elif "MixUp" in dataset_config["augmentation"]:
            mix = v2.RandomChoice(
                [v2.MixUp(num_classes=model_config["out_classes"]), v2.Identity()],
                p=[0.5, 0.5],
            )
        elif "CutMix" in dataset_config["augmentation"]:
            mix = v2.RandomChoice(
                [v2.CutMix(num_classes=model_config["out_classes"]), v2.Identity()],
                p=[0.5, 0.5],
            )
        else:
            mix = v2.Identity()
    else:
        mix = v2.Identity()

    def calc_train_loss(
        model: torch.nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        model.train()
        img, label = batch
        img, label = img.to(device), label.to(device)
        img, label = mix(img, label)

        pred = model(img)
        loss = criterion(pred, label)
        return loss

    return calc_train_loss


def get_eval_fn(train_loader: DataLoader, test_loader: DataLoader):
    """Returns a function to calculate several evaluation metrics for a model on a test dataset

    Args:
        train_loader (DataLoader): DataLoader providing training data
        test_loader (DataLoader): DataLoader providing the test data
    """

    def eval_model(
        model: torch.nn.Module, device: torch.device
    ) -> tuple[float, float, float, float]:
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            entropy_sum = 0
            for input, target in test_loader:
                input = input.to(device)
                target = target.to(device)

                logits = model(input)
                predicted = logits.argmax(dim=-1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                entropy_sum += (
                    torch.distributions.Categorical(logits=logits)
                    .entropy()
                    .sum()
                    .item()
                )
            test_accuracy = correct / total
            test_entropy = entropy_sum / total

            correct = 0
            total = 0
            entropy_sum = 0

            for input, target in train_loader:
                input = input.to(device)
                target = target.to(device)

                logits = model(input)
                predicted = logits.argmax(dim=-1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                entropy_sum += (
                    torch.distributions.Categorical(logits=logits)
                    .entropy()
                    .sum()
                    .item()
                )
            train_accuracy = correct / total
            train_entropy = entropy_sum / total

        return train_accuracy, train_entropy, test_accuracy, test_entropy

    return eval_model


def run(config: Config, cancel_at: int, device=torch.device("cpu"), early_stop=None):
    model_config: own_vit.ViTConfig = config["model"]
    train_config = config["train"]
    dataset_config = config["dataset"]

    model = own_vit.get_model(model_config)
    wandb.watch(model, log="all")
    model.to(device)
    if torch.cuda.get_device_capability(device)[0] >= 7:
        model.compile()
    optim = training_utils.get_optim(train_config, model)
    lr_scheduler = training_utils.get_scheduler(
        train_config, optim, d_model=model_config["d_model"]
    )
    train_loader, test_loader = dataset_utils.get_dataloader(dataset_config, "../data")

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=train_config["label_smoothing"]
    )

    calc_train_loss = get_loss_fn(model_config, criterion, dataset_config)
    eval_model = get_eval_fn(train_loader, test_loader)

    training_utils.run_training(
        model,
        train_config,
        optim,
        lr_scheduler,
        calc_train_loss,
        train_loader,
        eval_model,
        device,
        cancel_at=cancel_at,
        early_stop=early_stop,
    )


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser("train vit on cifar10")
    parser.add_argument("--cuda", type=int, default=-1)
    args = parser.parse_args()
    if args.cuda == -1:
        raise Exception("need to specify cuda number")
    device = torch.device("cuda:" + str(args.cuda))

    try:
        for change in variations:
            run_config = deep_update(base_config, change)  # type: ignore
            wandb.init(
                project="vit-classifier",
                config={
                    "dataset": run_config["dataset"],
                    "model": run_config["model"],
                    "train": run_config["train"],
                },
            )

            two_hours = int(time.time() + 2 * 60 * 60)
            try:
                run(wandb.config, two_hours, device=device)  # type: ignore
            except Exception as error:
                logger.exception(error)
    finally:
        import os

        os._exit(00)
