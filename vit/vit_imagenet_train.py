# %%
import logging
from typing import Any, Dict, TypeVar
import vit
import training_utils
import dataset_utils
import torch
import wandb
import argparse
from torchvision.models import VisionTransformer
from vit_github import ViT
import time
import torchvision.transforms.v2 as v2

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser("train vit on cifar10")
    parser.add_argument("--cuda", type=int, default=-1)
    args = parser.parse_args()
    if args.cuda == -1:
        raise Exception("need to specify cuda number")
    device = torch.device('cuda:' + str(args.cuda))

# %%
model_base_config: vit.ViTConfig = {
    'd_model': 192,
    'num_heads': 3,
    'num_layers': 12,
    'd_ffn': 3*192,
    'dropout': 0.1,
    
    'image_size': 128,
    'image_channels': 3,
    'patch_size': 16,
    'out_classes': 1_000,
    'conv_proj': True,
    'torch_att': True
}

train_base_config: training_utils.TrainConfig = {
    'num_steps': 1_00_000,
    'warmup_steps': 50_000,
    'optimizer': {
        'optim': "AdamW",
        'base_lr': 5e-4,
        'args': {
            'weight_decay': 2e-5
        }
    },
    'batches_per_step': 1,
    'eval_interval': 10_000,
    'log_interval': 1_000,
    'autocast': True,
    'lr_scheduler': "1cycle",
    'label_smoothing': 0.1,
    'clip_grad': None
}

dataset_base_config: dataset_utils.DatasetConfig = {
    'dataset': "IMAGENET",
    'augmentation': "AutoIMAGENET",
    'batch_size': 256,
    'num_workers': 2
}

base_config = {
    'model': model_base_config,
    'train': train_base_config,
    'dataset': dataset_base_config
}

variations = [
    {}
]

# %%
KeyType = TypeVar('KeyType')
def deep_update(mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping

def get_loss_fn(model_config: vit.ViTConfig, criterion, dataset_config: dataset_utils.DatasetConfig):
    if "MixUp" in dataset_config['augmentation'] and "CutMix" in dataset_config['augmentation']:
        mix = v2.RandomChoice([v2.MixUp(num_classes=model_config['out_classes']), v2.CutMix(num_classes=model_config['out_classes']), v2.Identity()], p=[0.25, 0.25, 0.5])
    elif "MixUp" in dataset_config['augmentation']:
        mix = v2.RandomChoice([v2.MixUp(num_classes=model_config['out_classes']), v2.Identity()], p=[0.5, 0.5])
    elif "CutMix" in dataset_config['augmentation']:
        mix = v2.RandomChoice([v2.CutMix(num_classes=model_config['out_classes']), v2.Identity()], p=[0.5, 0.5])
    else:
        mix = v2.Identity()
    
    def calc_train_loss(model, batch: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        model.train()
        img, label = batch
        img, label = img.to(device), label.to(device)
        img, label = mix(img, label)
        
        pred = model(img)
        loss = criterion(pred, label)
        return loss
    return calc_train_loss

def get_eval_fn(model_config: vit.ViTConfig, train_loader, test_loader):
    def eval_model(model, device) -> tuple[float, float, float, float]:
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
                entropy_sum += torch.distributions.Categorical(logits=logits).entropy().sum().item()
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
                entropy_sum += torch.distributions.Categorical(logits=logits).entropy().sum().item()
            train_accuracy = correct / total
            train_entropy = entropy_sum / total
            
        return train_accuracy, train_entropy, test_accuracy, test_entropy
    return eval_model

def run(config, cancel_at, device=torch.device('cpu')):
    model_config: vit.ViTConfig = config.model
    train_config = config.train
    dataset_config = config.dataset
    
    model = vit.get_model(model_config)
    model.to(device)
    if torch.cuda.get_device_capability(device)[0] >= 7:
        model.compile()
    optim = training_utils.get_optim(train_config, model)
    lr_scheduler = training_utils.get_scheduler(train_config, optim, d_model=model_config['d_model'])
    train_loader, test_loader = dataset_utils.get_dataloader(dataset_config, 'data')

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=train_config['label_smoothing'])
    
    calc_train_loss = get_loss_fn(model_config, criterion, dataset_config)
    eval_model = get_eval_fn(model_config, train_loader, test_loader)
    
    training_utils.train(
        model,
        train_config,
        optim,
        lr_scheduler,
        calc_train_loss,
        train_loader,
        eval_model,
        device,
        cancel_at=cancel_at
    )

# %%
if __name__ == '__main__':
    try:
        for change in variations:
            run_config = deep_update(base_config, change)
            wandb.init(project='vit-classifier', config={
                'dataset': run_config['dataset'],
                'model': run_config['model'],
                'train': run_config['train']
            }, notes="finally correct model")

            two_hours = time.time() + 18 * 60 * 60
            try:
                run(wandb.config, two_hours, device=device)
            except Exception as error:
                logger.exception(error)
    finally:
        import os
        os._exit(00)
