from vit_cifar_train import run
import torch
import wandb
import time
import argparse


sweep_conf = {
  "method": "bayes",
  "metric": {
    "goal": "maximize",
    "name": "accuracy"
  },
  "parameters": {
    "dataset": {
      "parameters": {
        "augmentation": {
          "value": "AutoCIFAR10-CutMix-MixUp"
        },
        "batch_size": {
          "value": 128
        },
        "dataset": {
          "value": "CIFAR10"
        },
        "num_workers": {
          "value": 8
        }
      }
    },
    "model": {
      "parameters": {
        "conv_proj": {
          "value": True
        },
        "d_ffn": {
          "distribution": "int_uniform",
          "max": 4096,
          "min": 512
        },
        "d_model": {
          "values": [128, 256, 384]
        },
        "dropout": {
          "value": 0.1
        },
        "image_channels": {
          "value": 3
        },
        "image_size": {
          "value": 32
        },
        "num_heads": {
          "values": [4, 8]
        },
        "num_layers": {
          "distribution": "int_uniform",
          "max": 12,
          "min": 2
        },
        "out_classes": {
          "value": 100
        },
        "patch_size": {
          "value": 4
        },
        "torch_att": {
          "value": True
        }
      }
    },
    "train": {
      "parameters": {
        "autocast": {
          "value": True
        },
        "batches_per_step": {
          "value": 1
        },
        "clip_grad": {
          "value": None
        },
        "eval_interval": {
          "value": 1000
        },
        "label_smoothing": {
          "value": 0.1
        },
        "log_interval": {
          "value": 200
        },
        "lr_scheduler": {
          "value": "warmup_cosine"
        },
        "num_steps": {
          "value": 60000
        },
        "optimizer": {
          "parameters": {
            "args": {
              "parameters": {
                "weight_decay": {
                  "value": 0.3
                }
              }
            },
            "base_lr": {
              "value": 0.001
            },
            "optim": {
              "value": "AdamW"
            }
          }
        },
        "warmup_steps": {
          "value": 3900
        }
      }
    }
  }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("train vit on cifar10")
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument("--sweep", required=True)
    args = parser.parse_args()
    if args.cuda == -1:
        raise Exception("need to specify cuda number")
    device = torch.device('cuda:' + str(args.cuda))
    wandb.login()

    def sweep_run():
        wandb.init(project="vit-classifier")
        cancel_at = time.time() + 60 * 60
        run(wandb.config, cancel_at, device=device)

    wandb.agent(args.sweep, project="vit-classifier", function=sweep_run, count=30)


