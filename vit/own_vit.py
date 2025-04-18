from typing import TypedDict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import math
from own_transformer import TransformerEncoderLayer


class ConvEmbedding(nn.Module):

    def __init__(self, channels: int, d_model: int, image_size: int, patch_size: int):
        """Init the convolutional embedding

        Args:
            channels (int): Channels of the input images
            d_model (int): Embedding dimension of the model
            image_size (int): Side-length of the squared images
            patch_size (int): Side-length of the squared patches
        """
        super(ConvEmbedding, self).__init__()

        self.d_model = d_model
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(
            channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Init weights
        fan_in = channels * patch_size * patch_size
        nn.init.trunc_normal_(self.conv.weight, std=math.sqrt(1 / fan_in))

    def forward(self, x):
        """Forward pass of the convolutional embedding

        B denots the batch size,
        C the image channels,
        H the image side-length.

        Args:
            x (Tensor): Input image of shape (B, C, H, H)

        Returns:
            Tensor: Embedded images of shape (B, K, d_model)
        """
        # [batch_size, channels, image_size, image_size]
        x = self.conv(x)
        # [batch_size, d_model, image_size / patch_size, image_size / patch_size]
        x = x.view(-1, self.d_model, self.num_patches).permute(0, 2, 1)
        # [batch_size, num_patches, d_model]
        return x


class OwnEmbedding(nn.Module):
    def __init__(self, channel: int, patch_size: int, d_model: int):
        """Init the self-implemented patch embedding

        Args:
            channel (int): Channels of the input images
            patch_size (int): Side-length of the squared patches
            d_model (int): Embedding dimension of the model
        """
        super(OwnEmbedding, self).__init__()
        self.channel = channel
        self.patch_size = patch_size
        flattened_patch_size = patch_size**2 * channel
        self.linear_proj = nn.Linear(flattened_patch_size, d_model, bias=False)

    def forward(self, x):
        """Forward pass of the self-implemented patch embedding

        B denots the batch size,
        C the image channels,
        H the image side-length.

        Args:
            x (Tensor): Input image of shape (B, C, H, H)

        Returns:
            Tensor: Embedded images of shape (B, K, d_model)
        """

        batch_size, channel, _, _ = x.size()

        patches = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .reshape(batch_size, channel, -1, self.patch_size**2)
        )
        flattened_patches = patches.permute(0, 2, 3, 1).flatten(-2, -1)
        return self.linear_proj(flattened_patches)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        d_model: int,
        image_size: int,
        patch_size: int,
        channels: int,
        num_heads: int,
        d_ffn: int,
        num_encoder_layer: int,
        num_classes: int,
        p_dropout: float = 0.1,
        conv_embedding: bool = False,
        torch_attention: bool = False,
    ):
        """Init the self-implemented VisionTransformer architecture

        Args:
            d_model (int): Embedding dimension of the model
            image_size (int): Side-length of the square-shaped input images
            patch_size (int): Side-length of the patches used for embedding the images
            channels (int): Channels of the input images
            num_heads (int): Number of heads for MHA
            d_ffn (int): Hidden dimension of the FFN
            num_encoder_layer (int): Number of encoder layers
            num_classes (int): Number of output classes for the classification head
            p_dropout (float, optional): Probability for Dropout. Defaults to 0.1.
            conv_embedding (bool, optional): If true, a convolutional embedding is used. If false, the self-implemented patch embedding is used. Defaults to False.
            torch_attention (bool, optional): If true, the PyTorch implementation of MHA is used for speedup through FlashAttention. Defaults to False.
        """
        super(VisionTransformer, self).__init__()

        num_patches = image_size**2 // patch_size**2
        if conv_embedding:
            self.patch_embedding = ConvEmbedding(
                channels, d_model, image_size, patch_size
            )
        else:
            self.patch_embedding = OwnEmbedding(channels, patch_size, d_model)

        self.pos_embedding = nn.Parameter(
            torch.empty(1, num_patches + 1, d_model).normal_(std=0.02)
        )
        self.embed_dropout = nn.Dropout(p_dropout)

        self.encoder = nn.Sequential(
            *(
                TransformerEncoderLayer(
                    d_model, num_heads, d_ffn, p_dropout, torch_attention
                )
                for _ in range(num_encoder_layer)
            )
        )
        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.class_token = nn.Parameter(torch.randn((d_model)))
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Init all weights with glorot initialization and biases with 0"""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_wd_params(self) -> tuple[set[str], set[str]]:
        """Return two parameter groups. The first group contains all parameters to which weigth decay may be applied,
        the second group contains the parameters where weigth decay must not be applied.

        Returns:
            Tuple: Weight decay parameter names, no weight decay parameter names
        """
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear, nn.Conv2d, nn.MultiheadAttention)
        modules_no_weight_decay = (nn.LayerNorm,)

        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = f"{m_name}.{param_name}" if m_name else param_name

                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)
                elif isinstance(m, VisionTransformer) and param_name in [
                    "class_token",
                    "pos_embedding",
                ]:
                    parameters_no_decay.add(full_param_name)

        assert len(parameters_decay & parameters_no_decay) == 0, (
            parameters_decay & parameters_no_decay
        )
        assert len(parameters_decay) + len(parameters_no_decay) == len(
            list(self.parameters())
        )
        return parameters_decay, parameters_no_decay

    def forward(self, imgs):
        """Forward pass of a VisionTransformer for image classification

        B denots the batch size,
        C the image channels,
        H the image side-length,
        N the number of classes

        Args:
            imgs (Tensor): Input images of shape (B, C, H, H)

        Returns:
            Tensor: Class logits of shape (B, N)
        """
        batch_size = imgs.size(0)

        embed = self.patch_embedding(imgs)
        class_tokens = (
            self.class_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        )
        embed = torch.cat((class_tokens, embed), dim=1)
        embed = embed + self.pos_embedding

        embed = self.embed_dropout(embed)

        z = self.encoder(embed)

        class_token = z[:, 0]
        class_token = self.encoder_norm(class_token)

        return self.mlp_head(class_token)

    @torch.no_grad()
    def predict(self, imgs):
        """Inference with a VisionTransformer

        B denots the batch size,
        C the image channels,
        H the image side-length

        Args:
            imgs (Tensor): Tensor of shape (B, C, H, H)

        Returns:
            Tensor: Predicted class for each image with shape (B,)
        """
        prob = self.forward(imgs)
        pred = prob.argmax(dim=-1)
        return pred


class ViTConfig(TypedDict):
    image_size: int
    image_channels: int
    patch_size: int

    d_model: int
    num_heads: int
    d_ffn: int
    num_layers: int
    dropout: float

    out_classes: int
    conv_proj: bool
    torch_att: bool


def get_model(config: ViTConfig) -> VisionTransformer:
    """Build ViT model for given config object

    Args:
        config (ViTConfig): Config object

    Returns:
        VisionTransformer: ViT model
    """
    return VisionTransformer(
        d_model=config["d_model"],
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        channels=config["image_channels"],
        d_ffn=config["d_ffn"],
        num_encoder_layer=config["num_layers"],
        num_heads=config["num_heads"],
        num_classes=config["out_classes"],
        p_dropout=config["dropout"],
        conv_embedding=config["conv_proj"] if "conv_proj" in config else False,
        torch_attention=config["torch_att"] if "torch_att" in config else False,
    )


class VitAttentionExtractor:

    def __init__(self, model: VisionTransformer) -> None:
        """Class which hooks into the forward pass of the self-implemented attention of the ViT to extract attention scores.

        Args:
            model (VisionTransformer): The model to hook into
        """
        self.num_layers = len(model.encoder)
        self.reset()
        for idx, layer in enumerate(model.encoder):
            layer: TransformerEncoderLayer
            layer.self_attention.attention_layer.dropout.register_forward_hook(
                self._get_hook(idx)
            )

    def _get_hook(self, layer_idx):
        def hook(model, input, output):
            self.attentions[layer_idx] = output.detach()

        return hook

    def reset(self):
        self.attentions = [None] * self.num_layers

    def attention_result(self):
        return self.attentions

    def attention_rollout(self, head_combination="max", ignore_ratio=0.2):
        tokens = self.attentions[0].size(-1)
        attentions_heads_combined = []
        for att in self.attentions:
            if head_combination == "max":
                combined = att.max(dim=1)[0]
            elif head_combination == "mean":
                combined = att.mean(dim=1)
            else:
                raise Exception("Unsupported head combination")

            # drop ignore_ratio lowest attentions
            flat = combined.view(combined.size(0), -1)
            _, idx = torch.topk(flat, int(flat.size(-1) * ignore_ratio), -1, False)
            idx = idx[idx != 0]  # never drop cls_token
            flat[0, idx] = 0
            attentions_heads_combined.append(combined)

        rollout = torch.eye(tokens)
        for att in attentions_heads_combined:
            I = torch.eye(tokens)
            a = (att + I) / 2
            a = a / a.sum(dim=-1)  # normalize att + I
            rollout = a @ rollout

        # first row is information flow to cls token
        cls_row = rollout.squeeze(0)[0, 1:]
        # shape from 196 to 14x14
        n = int(cls_row.size(-1) ** 0.5)
        cls_row = cls_row.view(n, n) / cls_row.max()  # normalize
        return cls_row


def show_image_attention(
    image: torch.Tensor, attention: torch.Tensor, interpolate: bool = False
):
    """Visualize rolled-out attention scores for an image

    Args:
        image (torch.Tensor): Image fo shape (C, H, H)
        attention (torch.Tensor): Attention scores
        interpolate (bool, optional): Whether to resize and interpolate the attention scores or to simply fill the whole patch. Defaults to False.

    Returns:
        _type_: _description_
    """
    image_size = image.size(-1)
    attention_size = attention.size(-1)
    scale = int(image_size / attention_size)

    mask = cm.jet(attention)
    if interpolate:
        mask = cv2.resize(mask, (image_size, image_size))
    else:
        mask = mask.repeat(scale, axis=0).repeat(scale, axis=1)
    mask = mask[:, :, :-1]
    image_np = image.permute(1, 2, 0).numpy()

    output = mask + image_np
    output = output - output.min()
    output = output / output.max()
    plt.imshow(output)
    return output
