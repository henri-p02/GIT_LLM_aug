from own_git import get_caption_model, GITConfig
from utils_datasets import cut_caption, get_caption_dataset, DatasetConfig, build_dataloader
import wandb
import torch
import os
import logging
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils_tokenizer import get_tokenizer_from_vocab
sys.path.append('./metrics')
from cider import CIDER
import random
import json
import utils_load_models

logging.basicConfig(level=logging.INFO)

def main(rank: int, args):
    
    if len(args.cuda) > 1:
        raise Exception("No mulitprocessing available")
    else:
        device = torch.device(args.cuda[0])
        logger = logging.getLogger(__name__)
    logger.info(f"Using {device}")
    
    logger.debug("Retreiving model config from wandb")
    model_conf: GITConfig = utils_load_models.get_run_config(name=args.run)['model']
    
    logger.debug("Got config, building model")
    tokenizer = get_tokenizer_from_vocab(model_conf["vocab_size"])
    logger.debug(f"Model needs {tokenizer} tokenizer")
    
    model = get_caption_model(model_conf, tokenizer)
    model.to(device)
    model.compile()
    logger.debug("Built model")
    
    last_path, best_path = utils_load_models.get_checkpoint(args.run)
    if args.checkpoint == 'last':
        checkpoint_path = last_path
    else:
        checkpoint_path = best_path
        
    if not is_file(checkpoint_path):
        raise argparse.ArgumentTypeError(f"Checkpoint {checkpoint_path} not found")
    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model'])
    logger.debug("Loaded model weights")    
    
    logger.debug("Building dataloader")
    ds_conf: DatasetConfig =  {
        "dataset": "COCO-karpathy",
        "batch_size": 512,
        "num_workers": 16,
        "augmentation": None,
        "eval_batch_size": 512,
        "prefetch_factor": 3,
        "grouped": True
    }
    train_set, val_set = get_caption_dataset(ds_conf, '../data/', tokenizer, len(args.cuda))
    train_loader, val_loader = build_dataloader(ds_conf, train_set, val_set)
    logger.debug("Built dataloader")
    
    logger.info("Starting Inference Validation")
    model.eval()
    scorer = CIDER(4)
    
    tok_name = "bert" if len(tokenizer.get_vocab()) < 50_000 else "gpt"
    scorer.load_df(f'../metrics/coco-karpathy-val-df-{tok_name}.pkl')
    results = []
    for img, caps in tqdm(val_loader):
        img = img.to(device, non_blocking=True)
        if not args.test:
            with torch.no_grad():
                candidates: torch.Tensor = model.infer(img, selector=args.selector) #type: ignore
            candidates = cut_caption(candidates, tokenizer)
        else:
            candidates = torch.tensor([random.choice(refs) for refs in caps.tolist()])
        for cand, refs in zip(candidates, caps):
            cand = cand[cand != tokenizer.pad_token_id]
            refs = [(ref[ref != tokenizer.pad_token_id]).tolist() for ref in refs]
            
            results.append({
                'candidate': {
                    'tokens': cand.tolist(),
                    'raw': tokenizer.decode(cand, skip_special_tokens=True)
                },
                'refrences': [
                    {
                        'tokens': ref,
                        'raw': tokenizer.decode(ref, skip_special_tokens=True)
                    } for ref in refs
                ]
            })
            logger.debug(f"Adding sample: {cand.tolist()} - {refs}")
            scorer.add_sample(cand.tolist(), refs)
            
    cider_mean, cider_std = scorer.calc_score(silent=not args.v)
    logger.info(f"Cider Score: Mean {cider_mean}, Std {cider_std}")
    
    if not args.test and not args.nosave:
        logger.debug("Writing results to file")
        result_path = f"results/infer_{args.run}_{args.dataset}_{args.selector}.json"
        with open(result_path, 'w') as f:
                json.dump({
                    'images': results,
                    'cider': {
                        'mean': cider_mean,
                        'std': cider_std
                    }
                },f)
                
        logger.info(f"Saved results to {result_path}")      
        
    return cider_mean, cider_std

def is_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} not found")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Run Inference on whole validation dataset")
    parser.add_argument("--cuda", action='append', required=True, type=int)
    parser.add_argument("--checkpoint", type=str, default="last", choices=["last", "best"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", "--resume", type=str, action='append')
    group.add_argument("--sweep", type=str)
    parser.add_argument("--selector", type=str, default='max')
    parser.add_argument("-v", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()
    
    if args.v:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    if args.sweep is not None or len(args.run) > 1:
        if args.sweep is not None:
            logging.info("Running a sweep")
            ids = utils_load_models.get_sweep_runs(args.sweep, {"state": 'finished', "tags": "no_final_cider"})
        else:
            logging.info("Running multiple runs")
            ids = [utils_load_models.get_run_id(run_name) for run_name in args.run]
        logging.info(f"Got {len(ids)} models to evals")
        
        failed_ids = {}
        with logging_redirect_tqdm():
            for id in tqdm(ids):
                logging.info(f"####### Running for {id}")
                try:
                    api = wandb.Api()
                    run = api.run(id)
                    args.run = run.name
                    
                    cider_mean, cider_std = main(0, args)
                    
                    if not args.nosave:
                        run.summary["final_cider_mean"] = cider_mean
                        run.summary["final_cider_std"] = cider_std
                        run.summary.update()
                        logging.info(f"Saved result for {args.run} to wandb!")
                        if "no_final_cider" in run.tags:
                            run.tags.remove("no_final_cider")
                            run.update()
                    
                except Exception as e:
                    failed_ids[id] = e
                    logging.error(f"####### Failed run for {id}")
                    logging.error(e)
        
        if len(failed_ids.keys()) > 0:
            logging.error(f"Failed for {list(failed_ids.keys())}")
    else:
        args.run = args.run[0]
        if len(args.cuda) > 1:
            # no multiprocesses because one runs just takes 2min
            raise Exception("No multiprocessing available")
        else:
            main(0, args)