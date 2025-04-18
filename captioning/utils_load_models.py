import wandb
import os
from wandb.errors.errors import CommError

__all__ = [
    "get_run_id",
    "get_run_name",
    "get_run_config"
]

run_ids = {
    'caption-plasma': "hpoeche-team/git-coco/xutw7a3q",
    'caption-firebrand': "hpoeche-team/git-coco/r0phf0le",
    'caption-resonance': "hpoeche-team/git-coco/s9cek72p",
    'brisk-dragon-19': "hpoeche-team/git-coco/x0bihebm",
    'skilled-glade-22': "hpoeche-team/git-coco/7624ps27",
    'hardy-oath-36': "hpoeche-team/git-coco/psm7wdsf",
    'fiery-butterfly-39': "hpoeche-team/git-coco/5u5uvunp",
    'genial-dew-41': "hpoeche-team/git-coco/hz5l8dcm",
    'dry-surf-43': "hpoeche-team/git-coco/i4wzszdw",
    'northern-hill-44': "hpoeche-team/git-coco/n7nfiz6u",
    'lemon-disco-46': "hpoeche-team/git-coco/8iegdicf",
    'tempting-flower-52': "hpoeche-team/git-coco/5ii7phtr",
    'charmed-universe-54': "hpoeche-team/git-coco/bgc2yu08",
    'trim-dew-72': "hpoeche-team/git-coco/p41hlggu",
    'gallant-wildflower-70': "hpoeche-team/git-coco/dz819fex"
}

projects = [
    "hpoeche-team/git-coco",
    "hpoeche-team/git-hp-search"
]

run_names = {v: k for k, v in run_ids.items()}

names_with_share_embed_false = ['caption-plasma', 'caption-firebrand', 'caption-resonance', 'brisk-dragon-19', 'skilled-glade-22', 'hardy-oath-36', 'fiery-butterfly-39', 'genial-dew-41', 'dry-surf-43', 'northern-hill-44', 'lemon-disco-46', 'tempting-flower-52', 'charmed-universe-54']

def get_run_id(name: str) -> str:
    try:
        return run_ids[name]
    except KeyError:
        api = wandb.Api()
        for project in projects:
            cands = api.runs(project, filters={'displayName': name})
            if len(cands) == 1:
                return '/'.join(cands[0].path)
        
        raise Exception(f"Name \"{name}\" is not valid")
                

def get_run_name(id: str) -> str:
    
    try:
        name = run_names[id]
    except KeyError:
        api = wandb.Api()
        try:
            run = api.run(id)
            name = run.name
        except CommError:
            raise Exception(f"Id \"{id}\" is not valid")
        
    return name
        
        

def get_run_config(id:str = None, name: str= None):
    if id is None and name is None:
        raise ValueError("Either id or name must be provided")
    if id is None:
        id = get_run_id(name)
    if name is None:
        name = get_run_name(id)
        
    api = wandb.Api()
    run = api.run(id)
    conf = run.config
    
    if name in names_with_share_embed_false:
        conf['model']['share_embedding'] = False
    
    return conf

def get_sweep_runs(id: str, filters: dict = dict()) -> list[str]:
    api = wandb.Api()
    
    project = "/".join(id.split("/")[:2])
    sweep_id = id.split("/")[2]
    all_filters = {
        "sweep": sweep_id,
        **filters
    }
    runs = api.runs(project, filters=all_filters, include_sweeps=True)
    
    return ['/'.join(run.path) for run in runs]

def get_checkpoint(name: str) -> tuple:
    fname = f"coco-{name}.pt"
    last_path = os.path.join('/home/poeche/ws/ba/code/captioning/models/', fname)
    best_path = os.path.join('/home/poeche/ws/ba/code/captioning/models/', f"best-{fname}")
    
    if not os.path.exists(last_path):
        last_path = None
    if not os.path.exists(best_path):
        best_path = None
        
    
    return last_path, best_path