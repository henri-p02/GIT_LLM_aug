from collections import defaultdict
from typing import Iterable, TypedDict, Union
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
from transformers import PreTrainedTokenizer
import numpy as np
from PIL import Image
import os
from functools import partial
import webdataset as wds
import torch.distributed as dist
import json
from itertools import islice
import random
import copy
from typing_extensions import NotRequired
from utils_tokenizer import get_tokenizer_from_vocab

DS_SIZES = {
    "CC3m": {"train": 2879488, "val": 13056},
}


class DatasetConfig(TypedDict):
    dataset: Union[str, list[str]]
    batch_size: int
    eval_batch_size: int
    num_workers: int
    prefetch_factor: int
    augmentation: Union[str, None]
    grouped: Union[bool, None]
    eval_dataset: NotRequired[Union[str, list[str], None]]


def visualize_images(
    images: torch.Tensor, scale=2, anns: Union[list[str], None] = None
):
    n = images.size(0)
    if anns is not None:
        assert len(anns) == n, "Number of images and anns must match"

    cols = round(math.sqrt(n))
    rows = math.ceil(n / cols)
    if anns is not None:
        rows = 2 * rows
    np_images = images.clone() - images.min()
    np_images /= np_images.max()

    fig, axs = plt.subplots(rows, cols, figsize=(scale * cols, scale * rows))
    if anns is not None:
        im_axs = axs[::2].flatten()
    else:
        im_axs = axs.flatten()
    for img, ax in zip(np_images, im_axs):
        ax.imshow(img.squeeze().detach().permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

    if anns is not None:
        an_axs = axs[1::2].flatten()
        for ann, ax in zip(anns, an_axs):
            ax.axis("off")
            t = ax.text(0, 0.5, ann, ha="left", wrap=True)
            t._get_wrap_line_width = lambda: 100

    plt.tight_layout()


def cut_caption(captions: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """Remove all tokens gerenated after the first EOS token and replace them with padding"""
    eos = tokenizer.eos_token_id
    pad = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad is None:
        raise Exception(f"No PAD or EOS token for tokenizer {tokenizer}")

    c2 = torch.cat(
        (
            captions,
            torch.ones((captions.size(0), 1), dtype=torch.int, device=captions.device)
            * eos,
        ),
        dim=-1,
    )
    lens = (c2[:, 1:] == eos).max(dim=-1)[1] + 2 # the first token can never be the end
    pad_mask = ~(
        torch.arange(captions.size(-1), device=captions.device) < lens[:, None]
    )
    captions[pad_mask] = pad
    return captions

# factory for defaultdict
def empty_list():
    return []


class CocoKarpathyDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        path,
        split,
        vocab_size,
        transform=None,
        tgt_transform=None,
        grouped=False,
        augmented="",
        world_size=1,
    ):
        """Dataset, loading the Karpathy split of MS-COCO. All captions consisting of more than 30 tokens are skipped.

        Args:
            path (str): Root path for datasets
            split (str): "train", "val", "test"
            vocab_size (int): Size of the vocabulary, also defining the used tokenizer.
            transform (optional): Transform, to apply to the images. Defaults to None.
            tgt_transform (optional): Transform to apply to the target captions. Defaults to None.
            grouped (bool, optional): If True, returns one image with all target captions.
                        If False, each caption is returned with an image individually. Defaults to False.
            augmented (str, optional): "llama" or "llama2" for the augmented datasets. Defaults to "".
            world_size (int, optional): If used for distributed training, the world size, to split the dataset. Defaults to 1.
        """
        super().__init__()
        self.split = split
        self.path = path
        self.grouped = grouped
        self.vocab_size = vocab_size
        self.augmented = augmented
        self.transform = transform
        self.tgt_transform = tgt_transform
        self.augmentation = defaultdict(empty_list)
        self.samples: Union[np.ndarray, None] = None
        self.world_size = world_size

        if augmented == "llama":
            self.group_size = 7
        elif augmented == "llama2":
            self.group_size = 10
        else:
            self.group_size = 5

    def lazy_init(self):
        with open(os.path.join(self.path, "karpathy/coco/dataset.json")) as f:
            images_array = json.load(f)["images"]
        self.samples = np.array(
            [img for img in images_array if img["split"] == self.split]
        )

        if self.augmented and self.split == "train":
            augment_meta = json.load(
                open(
                    os.path.join(
                        self.path, f"karpathy/coco/augment_{self.augmented}.json"
                    )
                )
            )["images"]
            self.augmentation = defaultdict(
                empty_list, {img["imgid"]: img["sentences"] for img in augment_meta}
            )

    def load_img(self, img_data):
        path = os.path.join(
            self.path, "coco/", img_data["filepath"], img_data["filename"]
        )
        raw_img = Image.open(path).convert("RGB")

        img = self.transform(raw_img) if self.transform is not None else raw_img
        return img

    def load_captions(self, img_data):
        sentences = img_data["sentences"] + self.augmentation[img_data["imgid"]]
        cap = [
            self.tokenizer(
                "<BOS>" + sentence["raw"] + "<EOS>",
                padding="max_length",
                max_length=30,
                add_special_tokens=False,
            )["input_ids"]
            for sentence in sentences
        ]
        cap = self.tgt_transform(cap) if self.tgt_transform is not None else cap
        return cap

    def single_iter(self, samples):
        for img_data in samples:
            img = self.load_img(img_data)
            for caption in self.load_captions(img_data):
                if len(caption) <= 30: # type: ignore
                    yield img, torch.tensor(caption)

    def grouped_iter(self, samples):
        for img_data in samples:
            img = self.load_img(img_data)
            captions = [cap for cap in self.load_captions(img_data) if len(cap) <= 30] # type: ignore
            if len(captions) < 5:
                captions = captions + captions[: 5 - len(captions)]
            yield img, torch.tensor(captions[: self.group_size])

    def __iter__(self):
        if self.samples is None:
            self.lazy_init()

        if dist.is_initialized() and self.world_size > 1:
            rank = dist.get_rank()

            samples = self.samples[rank :: self.world_size] #type:ignore
        elif "RANK" in os.environ and self.world_size > 1:
            rank = int(os.environ["RANK"])
            samples = self.samples[rank :: self.world_size] #type:ignore
        else:
            samples = self.samples

        info = torch.utils.data.get_worker_info()
        if info:
            samples = samples[info.id :: info.num_workers] #type:ignore
        else:
            samples = samples

        self.tokenizer = get_tokenizer_from_vocab(self.vocab_size)
        if self.grouped:
            yield from self.grouped_iter(samples)
        else:
            yield from self.single_iter(samples)


class RChainDataset(torch.utils.data.IterableDataset):
    """Class for combining two datasets, by picking batches from a weighted round robin algorithm"""
    def __init__(self, datasets, lens=None):
        super().__init__()
        self.datasets = datasets
        self.lens = lens

    def roundrobin(self, *iterables):
        pending = len(iterables)
        nexts = [iter(it).__next__ for it in iterables]
        while pending:
            next = random.sample(nexts, 1, counts=self.lens)[0]
            try:
                yield next()
            except StopIteration:
                pending -= 1
                nexts.remove(next)

    def __iter__(self):
        return self.roundrobin(*self.datasets)


def get_coco_karpathy_dataset(path, config: DatasetConfig, tokenizer, world_size):
    # mean and std of imagenet
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomResizedCrop(224, scale=(0.2, 1)),
    ] +
    augmentation_pipeline(config) + 
    [
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )

    tgt_transform = None

    grouped = (
        config["grouped"]
        if "grouped" in config and config["grouped"] is not None
        else False
    )
    
    llama_augmentation = (config["dataset"] + "-").split("-")[2] #type: ignore
    
    train_set = CocoKarpathyDataset(
        path,
        "train",
        len(tokenizer.get_vocab()),
        train_transform,
        tgt_transform,
        grouped=grouped,
        augmented=llama_augmentation,
        world_size=world_size,
    )
    val_set = CocoKarpathyDataset(
        path,
        "val",
        len(tokenizer.get_vocab()),
        test_transform,
        tgt_transform,
        grouped=grouped,
        world_size=world_size,
    )

    return train_set, val_set

def augmentation_pipeline(config: DatasetConfig) -> list[v2.Transform]:
    stages: list[v2.Transform] = []
    if "augmentation" in config and config["augmentation"] is not None:
        if "flip" in config["augmentation"].lower():
            stages += [v2.RandomHorizontalFlip(0.5)]
        if "rotate" in config["augmentation"].lower():
            stages += [v2.RandomRotation(25)] # type: ignore
        if "equilize" in config["augmentation"].lower():
            stages += [v2.RandomEqualize(0.2)]
        if "perspective" in config["augmentation"].lower():
            stages += [v2.RandomPerspective(0.2, 0.5)]
    return stages


def _make_sample(sample, transform, tokenizer):
    img = sample["jpg"].convert("RGB")
    tgt = sample["txt"]

    img = transform(img)
    tgt = tokenizer(
        "<BOS>" + tgt + "<EOS>",
        padding="max_length",
        max_length=30,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].squeeze()
    if tgt.size(0) <= 30:
        return img, tgt
    else:
        return None


def get_webdataset(
    path: str, config: DatasetConfig, tokenizer: PreTrainedTokenizer, world_size
):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(224, scale=(0.2, 1)),
        ] +
        augmentation_pipeline(config) + 
        [
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )

    train_set = (
        wds.WebDataset(
            path + "cc3m/cc3m-train-0{000..575}.tar",
            resampled=False,
            shardshuffle=True,
            nodesplitter=wds.split_by_node,
        )
        .shuffle(1000)
        .decode("pil")
        .map(partial(_make_sample, transform=train_transform, tokenizer=tokenizer))
        .with_epoch(DS_SIZES["CC3m"]["train"] // world_size)
    )

    test_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )

    test_set = (
        wds.WebDataset(
            path + "cc3m/cc3m-validation-00{00..15}.tar",
            resampled=False,
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
        )
        .decode("pil")
        .map(partial(_make_sample, transform=test_transform, tokenizer=tokenizer))
        .with_epoch(DS_SIZES["CC3m"]["val"] // world_size)
    )

    return train_set, test_set


def build_dataloader(config: DatasetConfig, train_set, test_set):
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    test_batch_size = config["eval_batch_size"]
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        num_workers=8,
        prefetch_factor=config["prefetch_factor"],
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, test_loader


def get_caption_dataset(
    config: DatasetConfig, path: str, tokenizer: PreTrainedTokenizer, world_size: int
) -> tuple[torch.utils.data.IterableDataset, torch.utils.data.IterableDataset]:
    if isinstance(config["dataset"], (list, tuple)):
        # combined dataset
        train_sets = []
        val_sets = []
        for ds in config["dataset"]:
            newconf = copy.deepcopy(config)
            newconf.update({"dataset": ds})
            train, val = get_caption_dataset(newconf, path, tokenizer, world_size)
            train_sets.append(train)
            val_sets.append(val)
        return RChainDataset(train_sets), torch.utils.data.ChainDataset(val_sets)

    else:
        # get a single dataset
        if world_size < 1:
            world_size = 1
        else:
            config["batch_size"] = config["batch_size"] // world_size
            config["eval_batch_size"] = config["eval_batch_size"] // world_size

        # augmented datasets are called COCO-karpathy-llama & COCO-karpathy-llama2
        if config["dataset"][: len("COCO-karpathy")] == "COCO-karpathy":
            train_set, val_set = get_coco_karpathy_dataset(
                path, config, tokenizer, world_size
            )
        elif config["dataset"] == "CC3m":
            train_set, val_set = get_webdataset(path, config, tokenizer, world_size)
        else:
            raise NotImplementedError(f'dataset {config["dataset"]} not implemented!')

        if (
            "eval_dataset" in config
            and config["eval_dataset"] is not None
            and set(config["eval_dataset"]) != set(config["dataset"])
        ):
            val_config = copy.deepcopy(config)
            val_config["dataset"] = config["eval_dataset"]
            val_config.pop("eval_dataset")
            _, val_set = get_caption_dataset(val_config, path, tokenizer, world_size)

        return train_set, val_set
    
    
#### For Testing
class SlicedDataLoader(Iterable):
    def __init__(self, dataloader: DataLoader, stop: int) -> None:
        self.dataloader = dataloader
        self.stop = stop

    def __iter__(self):
        return islice(self.dataloader, self.stop)


def reduce_dl_size(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_train_batches: int,
    num_test_batches: int,
):

    reduced_train_loader = SlicedDataLoader(train_loader, num_train_batches)
    reduced_test_loader = SlicedDataLoader(test_loader, num_test_batches)

    return reduced_train_loader, reduced_test_loader