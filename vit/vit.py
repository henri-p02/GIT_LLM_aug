from typing import TypedDict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import math

def _image_to_patches(image, patch_size):
    if image.dim() == 3:
        image = image.unsqueeze(0)
        
    assert image.dim() == 4, "Need batch, channel, h, w"
    batch_size, channel, _, _ = image.size()
    
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).reshape(batch_size, channel, -1, patch_size**2)
    flattened_patches = patches.permute(0, 2, 3, 1).flatten(-2, -1)
    return flattened_patches

def show_patches(image_patches, channels, patch_size):
    patches = image_patches.unflatten(-1, (patch_size, patch_size, channels)).permute(0, 1, 4, 2, 3)
    batch_size = patches.size(0)
    rows = int(patches.size(1) ** (0.5))
    
    for image in patches:
        fig, axs = plt.subplots(rows, rows, figsize=(4, 4))
        axs = axs.flatten()
        for img, ax in zip(image, axs):
            img = img.permute(1, 2, 0).detach().cpu().numpy()
            ax.imshow(img)
            ax.axis('off')
            
        fig.tight_layout()
        
class FFN(nn.Module):
    
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super(FFN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x):
        return self.model(x)
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, p_dropout, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.sqrt_dk = d_k ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        ## input [batch_size, heads, seq_len, d_k]
        ## mask [batch_size, 0, seq_len, seq_len]
       
        score: torch.Tensor = (q @ k.transpose(2, 3)) / self.sqrt_dk
        
        if mask is not None:
            score.masked_fill_(mask == 0, -float('inf'))
            
        score = self.dropout(self.softmax(score))
        attention = score @ v
        
        return attention
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, p_dropout: float):
        super(MultiHeadAttention, self).__init__()
        
        self.heads = heads
        self.d_k = d_model // heads
        self.w_q = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_k = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_v = nn.Linear(d_model, self.d_k * heads, bias=False)
        self.w_o = nn.Linear(heads * self.d_k, d_model)
        
        self.attention_layer = ScaledDotProductAttention(p_dropout, self.d_k)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # input [batch_size, seq_len, d_model]
        # mask [batch_size, seq_len, seq_len]
        
        # 1. project linearly
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. store heads in extra dim to compute attention parallel
        # [batch_size, seq_len, heads, d_k]
        batch_size, seq_len, _ = q.size()
        q = q.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        
        
        # 3. calculate attention
        if mask is not None:
            mask = mask.unsqueeze(1) # add dim for heads
        attention = self.attention_layer(q, k, v, mask)
        
        # 4. concatenate results
        # [batch_size, seq_len, d_k * heads]
        o = attention.transpose(1, 2).reshape(batch_size, -1, self.heads * self.d_k)
        o = self.w_o(o)
        
        return o
        

class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, d_ffn, p_dropout, torch_attention = False, skip_ffn = False):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_att_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.torch_attention = torch_attention
        if torch_attention:
            self.self_attention = nn.MultiheadAttention(d_model, num_heads, p_dropout, batch_first=True)
        else:
            self.self_attention = MultiHeadAttention(d_model, num_heads, p_dropout)
        self.self_att_dropout = nn.Dropout(p_dropout)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.skip_ffn = skip_ffn
        self.ffn_layer = FFN(d_model, d_ffn) 

        
    def forward(self, x):
        # [batch_size, seq_len, d_model]
        
        # 1. layernorm
        z = self.self_att_norm(x)
        
        # 2. self attention
        if self.torch_attention:
            att, _ = self.self_attention(z, z, z, need_weights=False)
        else:
            att = self.self_attention(z, z, z)
        att = self.self_att_dropout(att)
        
        # 3. residual connection
        y = att + x
        
        # 4. FFN
        if self.skip_ffn:
            out = y
        else:
            out = self.ffn_layer(self.ffn_norm(y))
            out = out + y
        
        return out
    
class ConvEmbedding(nn.Module):
    
    def __init__(self, channels, d_model, image_size, patch_size):
        super(ConvEmbedding, self).__init__()
        
        self.d_model = d_model
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(channels, d_model, kernel_size=patch_size, stride=patch_size)
        
        fan_in = channels * patch_size * patch_size
        nn.init.trunc_normal_(self.conv.weight, std=math.sqrt(1 / fan_in))
        
    def forward(self, x):
        # [batch_size, channels, image_size, image_size]
        x = self.conv(x)
        # [batch_size, d_model, image_size / patch_size, image_size / patch_size]
        x = x.view(-1, self.d_model, self.num_patches).permute(0, 2, 1)
        # [batch_size, num_patches, d_model]
        return x
    
class OwnEmbedding(nn.Module):
    def __init__(self, channel, patch_size, d_model):
        super(OwnEmbedding, self).__init__()
        self.channel = channel
        self.patch_size = patch_size
        flattened_patch_size = patch_size ** 2 * channel
        self.linear_proj = nn.Linear(flattened_patch_size, d_model, bias=False)

    def forward(self, x):
        
        batch_size, channel, _, _ = x.size()
        
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).reshape(batch_size, channel, -1, self.patch_size**2)
        flattened_patches = patches.permute(0, 2, 3, 1).flatten(-2, -1)
        return self.linear_proj(flattened_patches)
        
        
class VisionTransformer(nn.Module):
    
    def __init__(self, d_model, image_size, patch_size, channels, num_heads, d_ffn, num_encoder_layer, num_classes, p_dropout = 0.1, conv_embedding = False, torch_attention = False, skip_ffn=False):
        super(VisionTransformer, self).__init__()
        
        num_patches = image_size ** 2 // patch_size ** 2
        if conv_embedding:
            self.patch_embedding = ConvEmbedding(channels, d_model, image_size, patch_size)
        else:
            self.patch_embedding = OwnEmbedding(channels, patch_size, d_model)
        
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches+1, d_model).normal_(std=0.02))
        self.embed_dropout = nn.Dropout(p_dropout)       
        
        self.encoder = nn.Sequential(*(
            TransformerEncoderLayer(d_model, num_heads, d_ffn, p_dropout, torch_attention, skip_ffn) for _ in range(num_encoder_layer)
        ))
        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.class_token = nn.Parameter(torch.randn((d_model)))
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 2*d_model),
            nn.GELU(),
            nn.Linear(2*d_model, num_classes)
        )
        
        
        self._init_weights()
    
    def _init_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                
    def get_wd_params(self):
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear,nn.Conv2d,nn.MultiheadAttention)
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
                elif isinstance(m, VisionTransformer) and param_name in ['class_token', 'pos_embedding']:
                    parameters_no_decay.add(full_param_name)
                
        assert len(parameters_decay & parameters_no_decay) == 0, parameters_decay & parameters_no_decay
        assert len(parameters_decay) + len(parameters_no_decay) == len(list(self.parameters()))
        return parameters_decay, parameters_no_decay
    
    def forward(self, imgs):

        batch_size = imgs.size(0)
        
        embed = self.patch_embedding(imgs)        
        class_tokens = self.class_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        embed = torch.cat((class_tokens, embed), dim=1)
        embed = embed + self.pos_embedding
        
        embed = self.embed_dropout(embed)
        
        z = self.encoder(embed)
        
        class_token = z[:, 0]
        class_token = self.encoder_norm(class_token)
        
        return self.mlp_head(class_token)
    
    @torch.no_grad()
    def predict(self, imgs):
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
    return VisionTransformer(
        d_model=config['d_model'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        channels=config['image_channels'],
        d_ffn=config['d_ffn'],
        num_encoder_layer=config['num_layers'],
        num_heads=config['num_heads'],
        num_classes=config['out_classes'],
        p_dropout=config['dropout'],
        conv_embedding=config['conv_proj'] if "conv_proj" in config else True,
        torch_attention=config['torch_att'] if "torch_att" in config else False,
        skip_ffn=config['skip_ffn'] if 'skip_ffn' in config else False
    )


class VitAttentionExtractor:
    
    def __init__(self, model: VisionTransformer) -> None:
        self.num_layers = len(model.encoder)
        self.reset()
        for idx, layer in enumerate(model.encoder):
            layer: TransformerEncoderLayer
            layer.self_attention.attention_layer.dropout.register_forward_hook(self._get_hook(idx))
            
    def _get_hook(self, layer_idx):
        def hook(model, input, output):
            self.attentions[layer_idx] = output.detach()
        return hook
    
    def reset(self):
        self.attentions = [None] * self.num_layers
    
    def attention_result(self):
        return self.attentions
    
    def attention_rollout(self, head_combination='max', ignore_ratio=0.2):
        tokens = self.attentions[0].size(-1)
        attentions_heads_combined = []
        for att in self.attentions:
            if head_combination == 'max':
                combined = att.max(dim=1)[0]
            elif head_combination == 'mean':
                combined = att.mean(dim=1)
            else:
                raise Exception("Unsupported head combination")
            
            # drop ignore_ratio lowest attentions
            flat = combined.view(combined.size(0), -1)
            _, idx = torch.topk(flat, int(flat.size(-1)*ignore_ratio), -1, False)
            idx = idx[idx != 0] # never drop cls_token
            flat[0, idx] = 0
            attentions_heads_combined.append(combined)

        rollout = torch.eye(tokens)
        for att in attentions_heads_combined:
            I = torch.eye(tokens)
            a = ((att + I) / 2)
            a = a / a.sum(dim=-1) # normalize att + I
            rollout = a @ rollout
            
        # first row is information flow to cls token
        cls_row = rollout.squeeze(0)[0, 1:]
        # shape from 196 to 14x14
        n = int(cls_row.size(-1) ** 0.5)
        cls_row = cls_row.view(n, n) / cls_row.max() #normalize
        return cls_row
    
def show_image_attention(image: torch.Tensor, attention: torch.Tensor, interpolate: bool = False):
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
