from typing import Literal, TypedDict, Union
from own_transformer import TransformerEncoderLayer, TransformerDecoderLayer
from transformers import CLIPVisionModel
from transformers import GPT2Tokenizer, PreTrainedTokenizer, GPT2Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GITConfig(TypedDict):
    image_encoder: Literal["CLIP/ViT-B-16", "CLIP/ViT-B-32"]

    d_model: int
    num_heads: int
    d_ffn: int
    num_layers: int
    dropout: float
    torch_attn: bool

    vocab_size: int
    max_seq_len: int
    share_embedding: bool
    gpt_embedding: Union[None, bool]
    init_embedding: Union[None, str]
    pos_encoding: Union[None, Literal["learned", "sin", "gpt"]]
    cross_attention: bool


DEFUALT_GIT_CONFIG: dict = {"init_embedding": None, "cross_attention": False}


def get_caption_model(conf: GITConfig, tokenizer: PreTrainedTokenizer, frozen=False):
    conf = conf.copy()
    conf.update(**DEFUALT_GIT_CONFIG)

    if conf["image_encoder"] == "CLIP/ViT-B-16":
        encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        img_feat_dim = encoder.vision_model.config.hidden_size
    else:
        raise NotImplementedError(
            f'Invalid image encoder type: {conf["image_encoder"]}'
        )

    if frozen:
        for param in encoder.parameters():
            param.requires_grad = False

    model = GITCaptioning(
        image_encoder=encoder,
        img_feat_dim=img_feat_dim,
        vocab_size=conf["vocab_size"],
        tokenizer=tokenizer,
        d_model=conf["d_model"],
        d_ffn=conf["d_ffn"],
        num_decoder_layer=conf["num_layers"],
        num_heads=conf["num_heads"],
        p_dropout=conf["dropout"],
        max_seq_len=conf["max_seq_len"],
        torch_attention=conf["torch_attn"],
        share_embedding=conf["share_embedding"],
        cross_attention=conf["cross_attention"],
    )

    if "gpt_embedding" in conf and conf["gpt_embedding"]:
        assert (
            len(tokenizer) == 50257
        ), "GPT Embeddings are for GPT Tokenizer with 50257 token"
        gpt = GPT2Model.from_pretrained("gpt2")
        model.word_embedding.embedding.weight = torch.nn.Parameter(
            gpt.wte.weight.clone(), requires_grad=False
        )

        if conf["share_embedding"]:
            model.output.weight = model.word_embedding.embedding.weight

    elif "init_embedding" in conf and conf["init_embedding"] == "gpt2":
        gpt = GPT2Model.from_pretrained("gpt2")
        wte = gpt.wte.weight.clone()
        expand = torch.nn.init.xavier_normal_(
            torch.empty((len(tokenizer) - wte.size(0), wte.size(1)))
        )
        new_wte = torch.cat((wte, expand), dim=0)
        model.word_embedding.embedding.weight = torch.nn.Parameter(
            new_wte, requires_grad=True
        )
        if conf["share_embedding"]:
            model.output.weight = model.word_embedding.embedding.weight

    if "pos_encoding" in conf:
        if conf["pos_encoding"] == "gpt":
            gpt = GPT2Model.from_pretrained("gpt2")
            model.word_embedding.positions.weight = torch.nn.Parameter(
                gpt.wpe.weight.clone(), requires_grad=False
            )
        elif conf["pos_encoding"] == "sin":
            # build sinusodial pos encodings
            pe = torch.zeros((conf["max_seq_len"], conf["d_model"]))
            pos = torch.arange(conf["max_seq_len"]).unsqueeze(1)
            term = pos / (
                10000 ** (torch.arange(0, conf["d_model"], 2) / conf["d_model"])
            )

            pe[:, 0::2] = torch.sin(term)
            pe[:, 1::2] = torch.cos(term)

            model.word_embedding.positions.weight = torch.nn.Parameter(
                pe, requires_grad=False
            )

    return model


class WordEmbedingAndPositions(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        p_dropout: float = 0.0,
        scale: float = 1.0,
    ):
        """Init WordEmbeddingAndPositions

        Args:
            vocab_size (int): Size of the vocabulary to embed
            max_len (int): Maximum sequence length
            d_model (int): Embedding dimension of the model
            p_dropout (float, optional): Probability for Dropout. Defaults to 0.0.
            scale (float, optional): Scaling factor, applied to the weights. Usefull for a shared embedding. Defaults to 1.0.
        """
        super(WordEmbedingAndPositions, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positions = nn.Embedding(max_len, d_model)
        self.register_buffer("pos_idx", torch.arange(max_len).unsqueeze(0))
        self.dropout = nn.Dropout(p_dropout)
        self.scale_embedding = scale

    def forward(self, x):
        embed = self.embedding(x) * self.scale_embedding
        pos_embed = self.positions(self.pos_idx[:, : x.size(-1)])

        embed = embed + pos_embed
        return self.dropout(embed)


class GITCaptioning(nn.Module):

    def __init__(
        self,
        image_encoder: nn.Module,
        img_feat_dim: int,
        vocab_size: int,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        d_model: int,
        num_decoder_layer: int,
        num_heads: int,
        d_ffn: int,
        p_dropout: float = 0.0,
        share_embedding: bool = False,
        torch_attention: bool = False,
        cross_attention: bool = False,
    ):
        """Init GIT model

        Args:
            image_encoder (nn.Module): Model used as image encoder
            img_feat_dim (int): Output dimension of the image encoder features
            vocab_size (int): Size of the text vocabulary
            tokenizer (PreTrainedTokenizer): Tokenizer class
            max_seq_len (int): Maximum sequence length
            d_model (int): Embedding dimension of the model
            num_decoder_layer (int): Number of decoder layers
            num_heads (int): Number of attention heads for MHA
            d_ffn (int): Hidden dimension of the FFN
            p_dropout (float, optional): Probability for Dropout. Defaults to 0.0.
            share_embedding (bool, optional): If True, the weights of the token embedding and
                the linear layer in the decoder head are shared. Defaults to False.
            torch_attention (bool, optional): If True, the PyTorch implementation of MHA is used to speedup
                thorugh FlashAttention. Defaults to False.
            cross_attention (bool, optional): If True, the image features are not concatenated with the text input,
                but an additional cross-attention layer is used to combine image and text data. Defaults to False.
        """
        super(GITCaptioning, self).__init__()

        self.d_model = d_model

        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("tokenizer need to specify bos and eos token")
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_len = max_seq_len
        self.image_encoder = image_encoder
        self.image_embed_dim = img_feat_dim
        self.cross_attention = cross_attention
        self.image_feat_proj = nn.Linear(self.image_embed_dim, d_model)
        self.image_feat_norm = nn.LayerNorm(self.image_embed_dim)

        sqrt_dmodel = d_model**0.5 if share_embedding else 1.0
        self.word_embedding = WordEmbedingAndPositions(
            vocab_size, max_seq_len, d_model, p_dropout, scale=sqrt_dmodel
        )

        self.cap_pre_norm = nn.LayerNorm(self.d_model)

        if not self.cross_attention:
            self.text_decoder = nn.ModuleList(
                (
                    TransformerEncoderLayer(
                        d_model, num_heads, d_ffn, p_dropout, torch_attention
                    )
                    for _ in range(num_decoder_layer)
                )
            )
        else:
            self.text_decoder = nn.ModuleList(
                (
                    TransformerDecoderLayer(
                        d_model, num_heads, d_ffn, p_dropout, torch_attention
                    )
                    for _ in range(num_decoder_layer)
                )
            )

        self.out_ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        if share_embedding:
            self.output.weight = self.word_embedding.embedding.weight

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.image_feat_proj.weight, 0, 0.02)
        nn.init.constant_(self.image_feat_proj.bias, 0)
        nn.init.constant_(self.image_feat_norm.bias, 0)
        nn.init.constant_(self.image_feat_norm.weight, 1)

        nn.init.constant_(self.output.bias, 0)
        nn.init.constant_(self.out_ln.bias, 0)
        nn.init.constant_(self.out_ln.weight, 1)

        nn.init.normal_(self.word_embedding.embedding.weight, mean=0, std=0.02)

        nn.init.constant_(self.cap_pre_norm.bias, 0)
        nn.init.constant_(self.cap_pre_norm.weight, 1)

        for layer in self.text_decoder:
            nn.init.xavier_normal_(
                layer.ffn_layer.model[0].weight, gain=len(self.text_decoder) ** -0.5
            )
            nn.init.xavier_normal_(
                layer.ffn_layer.model[3].weight, gain=len(self.text_decoder) ** -0.5
            )

    def freeze_embedding(self):
        for param in self.word_embedding.parameters():
            param.requires_grad = False

    def get_param_groups(self):
        """Split parameters of the model in three groups
                1. Parameters of the text decoder with weight decay
                2. Parameters of the text decoder without weight decay
                3. Parameters of the image encoder

        Returns:
            Tuple: Three sets, containing the parameters listed as above
        """
        params_decay = list()
        params_no_decay = list()

        params_image_encoder = list(self.image_encoder.parameters())

        params_no_decay.extend((p for p in self.word_embedding.parameters()))
        params_no_decay.append(self.image_feat_proj.bias)
        params_decay.append(self.image_feat_proj.weight)
        params_no_decay.extend((p for p in self.image_feat_norm.parameters()))
        params_no_decay.extend((p for p in self.out_ln.parameters()))

        params_no_decay.extend(
            (p for p in self.output.parameters() if p not in set(params_no_decay))
        )

        for layer in self.text_decoder:
            params_no_decay.extend((p for p in layer.self_att_norm.parameters()))
            params_no_decay.extend((p for p in layer.ffn_norm.parameters()))

            if self.cross_attention:
                layer_params = (
                    list(layer.self_attention.named_parameters())
                    + list(layer.cross_attention.named_parameters())
                    + list(layer.ffn_layer.named_parameters())
                )
                params_no_decay.extend((p for p in layer.cross_att_norm.parameters()))
            else:
                layer_params = list(layer.self_attention.named_parameters()) + list(
                    layer.ffn_layer.named_parameters()
                )

            for p_name, p in layer_params:
                if p_name.endswith("bias"):
                    params_no_decay.append(p)
                else:
                    params_decay.append(p)

        if hasattr(self, "cap_pre_norm"):
            params_no_decay.extend((p for p in self.cap_pre_norm.parameters()))

        return params_decay, params_no_decay, params_image_encoder

    def forward(self, image, caption):
        """Forward pass of the complete GIT model

        B = batch size,
        C = image channels,
        H = image side-length,
        L = maximum text sequence length,
        V = vocabulary size

        Args:
            image (Tensor): Input image of shape (B, C, H, H)
            caption (Tensor): Tokenized text of shape (B, L)

        Returns:
            Tensor: Token logits with shape (B, L, V)
        """

        # batch_size x channel x w x h
        image_feats = self.image_encoder(image).last_hidden_state
        # batch_size x 197 x d_model_vit
        num_image_feats = image_feats.size(-2)
        image_feats = self.image_feat_norm(self.image_feat_proj(image_feats))
        # batch_size x 197 x d_model

        # batch_size x max_len
        caption_embed = self.word_embedding(caption)
        caption_embed = self.cap_pre_norm(caption_embed)
        # batch_size x max_len x d_model

        if not self.cross_attention:
            transformer_input = torch.cat((image_feats, caption_embed), dim=1)
            # batch_size x (197 + max_len) x d_model

            # create masks
            device = transformer_input.device
            key_pad_mask = self._text_pad_mask(
                caption, image_feats.size(-2), device=device
            )
            att_mask = self._image_text_mask(
                caption.size(-1), image_feats.size(-2), device=device
            )

            # pass through decoder layers
            for layer in self.text_decoder:
                transformer_input = layer(
                    transformer_input, attn_mask=att_mask, key_pad_mask=key_pad_mask
                )
            # consider only output of text tokens
            text_out = transformer_input[:, num_image_feats:]
        else:
            # self.cross_attention == True
            transformer_input = caption_embed

            device = transformer_input.device
            key_pad_mask = self._text_pad_mask(caption, 0, device)
            att_mask = self._image_text_mask(caption.size(-1), 0, device)
            for layer in self.text_decoder:
                transformer_input = layer(
                    transformer_input,
                    image_feats,
                    attn_mask=att_mask,
                    key_pad_mask=key_pad_mask,
                )
            # decoder outputs only text tokens
            text_out = transformer_input

        logits = self.output(self.out_ln(text_out))
        return logits

    def _text_pad_mask(self, text_batch, num_im_token, device) -> torch.Tensor:
        """Builds the according mask for padding tokens on the text batch.
        Padding is not identified by the padding token, but everything after the first EOS token,
        which is not on position 0, is masked. This enables to use text data, which uses the EOS token,
        for marking begin, end and padding of a sequence (like GPT2).

        B = batch size
        L = maximum text sequence length,
        K = number of image features

        Args:
            text_batch (Tensor): Tensor of tokens with shape (B, L)
            num_im_token (int): Number of image tokens K
            device (torch.device): Device to crate the mask on

        Returns:
            torch.Tensor: Mask of shape (B, K+L) with True, where attention is not allowed
        """
        batch_size = text_batch.size(0)
        num_txt_token = text_batch.size(-1)
        total_token = num_txt_token + num_im_token

        # dont attend to all tokens after eos token (except first token bc bos=eos)
        eos_pos = text_batch[:, 1:] == self.eos_token_id  # first token might be bos=eos
        eos_pos = torch.cat(
            (eos_pos, torch.zeros((batch_size, 1), device=device)), dim=-1
        )  # pad eos_pos for case num_txt_token=1
        eos_idx = (
            eos_pos.max(dim=-1)[1] + 2
        )  # first occurance + 1 for bos + 1 to start after

        mask = torch.zeros((batch_size, total_token), dtype=torch.bool, device=device)
        mask[:, num_im_token:] = ~(
            torch.arange(num_txt_token, device=device) < eos_idx[:, None]
        )
        return mask

    def _image_text_mask(self, num_txt_token: int, num_im_token: int, device):
        """Create seq2seq mask, for concatenated image features and text tokens

        L = maximum text sequence length,
        K = number of image features

        Args:
            num_txt_token (int): Number of text tokens
            num_im_token (int): Number of image tokens
            device (torch.device): Device to create the mask on

        Returns:
            Tensor: Mask of shape (K+L, K+L) with True, where attention is not allowed
        """

        total_token = num_im_token + num_txt_token
        mask = torch.zeros((total_token, total_token), dtype=torch.bool, device=device)

        # image do not attend to text
        mask[:num_im_token, num_im_token:] = 1
        # text do not attend to future text
        mask[num_im_token:, num_im_token:] = ~torch.tril(
            torch.ones((num_txt_token, num_txt_token), dtype=torch.bool, device=device)
        )

        return mask

    def infer(
        self,
        image: torch.Tensor,
        selector="max",
        return_token_probs: Union[None, list[int]] = None,
        min_2_gram_dist: int = 0,
        prevent_repetition: bool = False,
    ):
        """Infer caption for input images.

        B = batch size,
        C = image channels,
        H = image side-length,
        L = maximum sequence length

        Args:
            image (torch.Tensor): Input images as Tensor of size (B, C, H, H)
            selector (str, optional): Sampling strategy. "max" selects the token with maximum probability as the nex token,
                                        "top-k-x" samples from the top x tokens, "top-p-x" samples from the tokens covering probability mass of x.
                                        Defaults to "max".
            return_token_probs (list[int], optional): List of tokens, for which the probailities for each sampling step are recorded and returned in the end. Defaults to None.
            min_2_gram_dist (int, optional): If greater than 0, token logits are masked, to prevent generation of all bigrams appeared in the last min_2_gram_dist tokens. Defaults to 0.
            prevent_repetition (bool, optional): If True, the direct repitition of a token is prevented. Defaults to False.

        Returns:
            Tensor: Tensor with the inferred captions of shape (B, L). If return_token_probs is present,
                    the return value is a tuple of the captions and a Tensor of shape (B, L, N) containing the probabilites
                    for all N tokens, for each generation step.
        """
        device = image.device
        with torch.no_grad():
            caps = torch.empty(
                (image.size(0), 1), dtype=torch.int, device=device
            ).fill_(self.bos_token_id)

            if return_token_probs is not None:
                requested_token_probs = torch.zeros(
                    (image.size(0), len(return_token_probs), 1), device=device
                )

            with torch.profiler.record_function("image_encoder_forward"):
                image_feats = self.image_encoder(image).last_hidden_state
                image_feats = self.image_feat_proj(self.image_feat_norm(image_feats))
                num_image_feats = image_feats.size(-2)

            while caps.size(-1) < self.max_seq_len:

                with torch.profiler.record_function("text_decoder_forward"):
                    caption_embed = self.word_embedding(caps)
                    key_pad_mask = self._text_pad_mask(
                        caps, image_feats.size(-2), device=device
                    )
                    att_mask = self._image_text_mask(
                        caps.size(-1), image_feats.size(-2), device=device
                    )

                    transformer_input = torch.cat((image_feats, caption_embed), dim=1)
                    for layer in self.text_decoder:
                        transformer_input = layer(
                            transformer_input,
                            attn_mask=att_mask,
                            key_pad_mask=key_pad_mask,
                        )
                    text_out = transformer_input[:, num_image_feats:]
                    logits = self.output(self.out_ln(text_out))
                    logits = logits[:, -1:]
                    # shape: batch_size x 1 x vocab_size

                with torch.profiler.record_function("sampling"):
                    # modify logits to avoid 2-gram repetition
                    if min_2_gram_dist > 0 and caps.size(-1) >= 2:
                        last_token = caps[:, -1:]
                        idx_last_token = torch.where(caps == last_token)
                        mask = torch.logical_and(
                            idx_last_token[1] >= caps.size(-1) - 2 - min_2_gram_dist,
                            idx_last_token[1] < caps.size(-1) - 1,
                        )
                        cap_idx = idx_last_token[0][mask]
                        token_idx = caps[cap_idx, idx_last_token[1][mask] + 1]

                        forbidden = torch.zeros_like(logits, dtype=torch.bool)
                        forbidden[cap_idx, :, token_idx] = True
                        forbidden[cap_idx, :, self.eos_token_id] = False

                        logits[forbidden] = -float("inf")

                    if prevent_repetition and caps.size(-1) >= 1:
                        last_token = caps[:, -1:]
                        forbidden = torch.zeros_like(logits, dtype=torch.bool)
                        forbidden[:, :, last_token] = True
                        forbidden[:, :, self.eos_token_id] = False

                        logits[forbidden] = -float("inf")

                    if return_token_probs is not None:
                        token_probs = torch.nn.functional.softmax(logits, dim=-1)[
                            :, -1, return_token_probs
                        ]
                        requested_token_probs = torch.cat(
                            (requested_token_probs, token_probs[:, :, None]), dim=2  # type: ignore
                        )

                    if selector == "max":
                        preds = logits.argmax(dim=-1)  # prediction for last token
                    elif selector == "sample":
                        preds = torch.distributions.Categorical(logits=logits).sample()
                    elif selector == "top5":
                        top_logits = logits.topk(5, dim=-1)
                        idx_sample = torch.distributions.Categorical(
                            logits=top_logits.values
                        ).sample()
                        preds = top_logits.indices.squeeze(1).gather(
                            dim=-1, index=idx_sample
                        )
                    elif selector[: len("top-p-")] == "top-p-":
                        p = int(selector[len("top-p-") :]) / 100
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                        v_p_mask = sorted_probs.cumsum(dim=-1) <= p

                        # shit right to be above p and always take first token
                        v_p_mask = v_p_mask.roll(1, dims=-1)
                        v_p_mask[:, :, 0] = True

                        sorted_probs[~v_p_mask] = 0  # cancel all token not in nucleus
                        sorted_probs /= sorted_probs.sum(dim=-1)[
                            ..., None
                        ]  # scale probs
                        probs = sorted_probs.gather(
                            dim=-1, index=sorted_idx.argsort(-1)
                        )  # reverse sorting

                        preds = torch.distributions.Categorical(probs=probs).sample()
                    else:
                        raise NotImplementedError()
                    caps = torch.cat((caps, preds), dim=-1)  # append prediction

        if return_token_probs is not None:
            return caps, requested_token_probs  # type: ignore
        else:
            return caps
