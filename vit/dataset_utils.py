from pathlib import Path
from typing import Any, Callable, Literal, TypedDict, Union
import torchvision
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

class DatasetConfig(TypedDict):
    dataset: str
    batch_size: int
    num_workers: int
    augmentation: Union[str, None]
    
    
def visualize_images(images: torch.Tensor):
    n = images.size(0)
    cols = round(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img.squeeze().detach().permute(1, 2, 0).cpu().numpy())
        ax.axis('off')

    # delete empty plots
    [fig.delaxes(ax) for ax in axs if not ax.has_data()]

    plt.tight_layout()
    
    
def get_dataloader_cifar(config: DatasetConfig, path: str) -> tuple[DataLoader, DataLoader]:
    train_data = None
    test_data = None
    
    norm = v2.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
    
    test_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            norm
        ])
    
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        norm
    ])
    if config["augmentation"] is not None and "AutoCIFAR10" in config['augmentation']:
        train_transform = v2.Compose([
            train_transform,
            v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)
        ])
    
    
    if config['dataset'] == "CIFAR10":
        train_data = torchvision.datasets.CIFAR10(path, download=True, train=True, transform=train_transform)
        test_data = torchvision.datasets.CIFAR10(path, download=True, train=False, transform=test_transform)
    elif config['dataset'] == "CIFAR100":
        train_data = torchvision.datasets.CIFAR100(path, download=True, train=True, transform=train_transform)
        test_data = torchvision.datasets.CIFAR100(path, download=True, train=False, transform=test_transform)
        
    if train_data is not None and test_data is not None:
        train_loader = DataLoader(train_data, config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        test_loader = DataLoader(test_data, config['batch_size'], shuffle=False)
    else:
        raise Exception("No valid dataset")
    

    return train_loader, test_loader


class ImageNetDataset(torchvision.datasets.ImageFolder):
    
    def __init__(self, root: str, split: Literal["train", "val"] = "train", transform = None, target_transform = None):
        path = root + '/' + split
        super().__init__(path, transform, target_transform)

def get_dataloader(config: DatasetConfig, path: str) -> tuple[DataLoader, DataLoader]:
    if config['dataset'] in ["CIFAR10", "CIFAR100"]:
        return get_dataloader_cifar(config, path)
    
    if config['dataset'] == "IMAGENET":
        normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        augment = v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET) if config['augmentation'] is not None else v2.Identity()
        
        train_transform = v2.Compose([
            v2.RandomResizedCrop(128),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
            augment
        ])
        
        test_transform =v2.Compose([
            v2.Resize(128),
            v2.CenterCrop(128),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize
        ])
        
        train_data = ImageNetDataset(path + "/ILSVRC/Data/CLS-LOC", split="train", transform=train_transform)
        test_data = ImageNetDataset(path + "/ILSVRC/Data/CLS-LOC", split="val", transform=test_transform)
        
        train_loader = DataLoader(train_data, config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        test_loader = DataLoader(test_data, config['batch_size'], shuffle=False)
        
    return train_loader, test_loader