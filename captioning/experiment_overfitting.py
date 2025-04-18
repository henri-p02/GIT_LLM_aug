from transformers import BertModel
from typing import TypedDict
from utils_datasets import DatasetConfig, cut_caption, get_tokenizer_from_vocab
from own_git import GITConfig, get_caption_model, GITCaptioning
from utils_training import TrainConfig, get_optim, get_scheduler
import torch
import torch.nn as nn
import argparse
import logging
import json
from torchvision.transforms import v2
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import DataLoader

T=torch.Tensor

class Config(TypedDict):
    model: GITConfig
    dataset: DatasetConfig
    train: TrainConfig

config: Config = {
    'model': {
        'cross_attention': False,
        'image_encoder': 'CLIP/ViT-B-16',
        'vocab_size': 28999,
        'max_seq_len': 30,
        'share_embedding': True,
        
        'd_model': 768,
        'd_ffn': 2*768,
        'num_layers': 6,
        'num_heads': 12,
        'dropout': 0.1,
        'torch_attn': True,
        'init_embedding': None,
        'gpt_embedding': False,
        'pos_encoding': 'learned'
    },
    'dataset': {
        'dataset': 'single-img',
        'batch_size': 1,
        'eval_batch_size': 0,
        'num_workers': 0,
        'prefetch_factor': 0,
        'augmentation': '',
        'grouped': False
    },
    'train': {
        'resulting_batch_size': 1,
        'num_steps': 10_000,
        'warmup_steps': 500,
        'optimizer': {
            'optim': "AdamW",
            'base_lr': 5e-5,
            'args': {
                'weight_decay': 0.2,
                'betas': (0.9, 0.99)
            }
        },
        'lr_scheduler': 'warmup_cosine',
        'autocast': True,
        'clip_grad': 100000,
        'batches_per_step': 1,
        'log_interval': 10,
        'eval_interval': 200_000,
        'label_smoothing': 0.1,
        'early_stopping': 0
    }
}

tokenizer = get_tokenizer_from_vocab(config['model']['vocab_size'])

def get_one_batch_example(batch_size, tokenizer, fixed_ids=None):
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(224, scale=(0.1, 1)),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(25), # type: ignore
            v2.RandomEqualize(0.2),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )

    data = COCO("../data/coco/annotations/captions_train2014.json")
    ids = list(data.anns.keys())

    captions = []
    images = torch.empty((0, 3, 224, 224))
    output_ids = []
    i = 0
    while images.size(0) < batch_size:
        if fixed_ids is None:
            pos = torch.randint(0, len(ids), (1,)).item()
        else:
            pos = fixed_ids[i]
            i += 1
        cap = tokenizer.bos_token + data.anns[ids[pos]]["caption"] + tokenizer.eos_token # type: ignore
        if len(tokenizer(cap)["input_ids"]) <= 30:
            captions.append(cap)
            output_ids.append(pos)
            img_name = data.imgs[data.anns[ids[pos]]["image_id"]]["file_name"] # type: ignore
            img = Image.open(f"../data/coco/train2014/{img_name}").convert("RGB")
            img = transform(img)
            images = torch.cat((images, img.unsqueeze(0)), dim=0)

    captions = torch.tensor(
        tokenizer(captions, padding=True, add_special_tokens=False)["input_ids"]
    )
    dataset = torch.utils.data.TensorDataset(images, captions)
    print(output_ids)
    return DataLoader(dataset, batch_size)


def calc_train_loss_git(model: nn.Module, batch: list[T], device: torch.device) -> T:
    model.train()
    im, caption = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    
    logits: T = model(im, caption)
    # batch_size x seq_len x vocab_size
    
    tgt = caption[:, 1:]
    logits = logits[:, :-1]
    
    eos_pos = tgt == tokenizer.eos_token_id
    lens = eos_pos.max(dim=-1)[1] + 1
    pad_mask = (torch.arange(tgt.size(-1), device=device) < lens[:, None])
    
    tgt = tgt[pad_mask]
    logits = logits[pad_mask]
    
    loss = criterion(logits, tgt)
    return loss
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("train a git on a dataset")
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument('--init_embed', action='store_true')
    args = parser.parse_args()
    if args.cuda == -1:
        raise Exception("need to specify cuda number")
    device = torch.device('cuda:' + str(args.cuda))
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.debug('Building model...')
    model = get_caption_model(config['model'], tokenizer)
    logger.debug('Model built')
    
    if args.init_embed:
        bert = BertModel.from_pretrained('bert-base-cased')
        wte = bert.embeddings.word_embeddings.weight
        expand = torch.nn.init.xavier_normal_(torch.empty((len(tokenizer) - wte.size(0), wte.size(1))))
        new_wte = torch.cat((wte, expand), dim=0)
        model.word_embedding.embedding.weight = torch.nn.Parameter(new_wte, requires_grad=True)

    logger.debug('Building dataloader...')
    with open('overfit_results/ids.json') as f:
        ids = json.load(f)

    train_loader = get_one_batch_example(256,tokenizer,fixed_ids=ids)
    logger.debug('Dataloader built')

    logger.debug('Building optimizer and scheduler...')
    optim = get_optim(config['train'], model)
    lr_scheduler = get_scheduler(config['train'], optim)
    logger.debug('Optimizer and scheduler built')

    criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'])

    model.to(device)
    if torch.cuda.get_device_capability(device)[0] >= 7:
        model.compile()
    scaler = torch.amp.grad_scaler.GradScaler(device=device.type)
    
    losses = []
    
    try:
        batch = next(iter(train_loader))
        batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
        logger.error(batch[1].shape)
        logger.error('\n'.join(tokenizer.batch_decode(cut_caption(batch[1], tokenizer), skip_special_tokens=True)))
        
        for step in range(config['train']['num_steps']):
            optim.zero_grad(True)
            with torch.autocast(device.type):
                model.train()
                loss = calc_train_loss_git(model, batch, device)
                
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            lr_scheduler.step()
            
            if step % 100 == 0:
                model.eval()
                with torch.autocast(device.type):
                    cap: torch.Tensor = model.infer(batch[0], selector='max') # type: ignore
                    cut_cap = cut_caption(cap, tokenizer)
                                        
                cut_cap = cut_cap.to(device)
                if cut_cap.size(1) < batch[1].size(1):
                    cut_cap = torch.cat([cut_cap, torch.ones((cut_cap.size(0), batch[1].size(1) - cut_cap.size(1))) * tokenizer.pad_token_id], dim = -1)
                equals = cut_cap[:, :batch[1].size(1)] == batch[1]
                accuracy = equals.to(dtype=float).mean().item()
                losses.append({'step': step, 'loss': loss.item(), 'acc': accuracy})
                logger.info(f"Step: {step:6d}/{config['train']['num_steps']:6d}, Loss: {loss.item():.10f}, Accuracy: {accuracy:.2f}")#
                
                
                decoded = tokenizer.batch_decode(cut_cap, skip_special_tokens=False)
                for i in range(len(decoded)):
                    if equals.min(dim=-1)[0][i]:
                        logger.debug(decoded[i])
                    else:
                        logger.debug(decoded[i] + " <-> " + tokenizer.decode(cut_caption(batch[1][i].unsqueeze(0), tokenizer)[0], skip_special_tokens=False))
                if equals.min():
                    logger.error("##### Got all right! ######")
            
            
    except Exception as error:
        logger.exception(error)
    finally:
        with open('overfit_results/00result.json', "w") as f:
            f.write(json.dumps(losses))
        import os
        os._exit(00)