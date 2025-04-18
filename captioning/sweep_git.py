import wandb
import argparse
import train_git
from functools import partial
import torch.multiprocessing as mp

def run_training(args):
    wandb_run = wandb.init()
    config = wandb_run.config
    
    # calc train set size
    if config['dataset']['dataset'] == 'COCO-karpathy':
        train_size = 412672
    elif config['dataset']['dataset'] == 'COCO-karpathy-llama':
        train_size = 412672 * 1.4
    elif config['dataset']['dataset'] == 'COCO-karpathy-llama2':
        train_size = 412672 * 2
    else:
        train_size = 400000

    # eval every 2 epochs
    config['train']['eval_interval'] = (train_size // config['train']['resulting_batch_size']) * 2
    
    train_git.run(0, args, config, wandb_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='append', required=True, type=int)
    parser.add_argument("--sweep_id", "--s", required=True, type=str)
    parser.add_argument("--counts", "--c", required=True, type=int)
    args = parser.parse_args()

    args.test = False
    args.resume = None
    args.checkpoint = None
    args.sweep = True

    wandb.agent(args.sweep_id, function=partial(run_training, args=args), count=args.counts)