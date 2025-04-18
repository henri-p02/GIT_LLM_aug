import matplotlib.pyplot as plt
from matplotlib import colors
import torch

def plot_pos_embed(embed_values: torch.Tensor, patches_per_row: int):
    patch_embed = embed_values / embed_values.norm(dim=1).unsqueeze_(-1)
    cos_sim = patch_embed @ patch_embed.T
    cos_sim = cos_sim.abs().cpu()
    fig, axs = plt.subplots(patches_per_row, patches_per_row)

    norm = colors.Normalize(vmin=cos_sim.min().item(), vmax=cos_sim.max().item())

    for i, ax in enumerate(axs.flat):
        im = ax.imshow(cos_sim[i].view(patches_per_row, patches_per_row), norm=norm)
        ax.axis('off')
        
    fig.colorbar(im, ax=axs.ravel().tolist())