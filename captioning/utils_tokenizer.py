from transformers import GPT2TokenizerFast as GPT2Tokenizer
from transformers import BertTokenizerFast as BertTokenizer
from transformers import PreTrainedTokenizer


def get_tokenizer(type: str, original=False) -> PreTrainedTokenizer:
    if type == "gpt2":
        if not original:
            tokenizer = GPT2Tokenizer.from_pretrained(
                "gpt2",
                pad_token="<PAD>",
                bos_token="<BOS>",
                eos_token="<EOS>",
                unk_token="<UNK>",
            )
        else:
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif type == "bert-cased":
        tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-base-cased",
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
        )
    else:
        raise NotImplementedError(f"Invalid tokenizer type: {type}")
    return tokenizer


def get_tokenizer_from_vocab(vocab_size: int) -> PreTrainedTokenizer:
    vocab_sizes = {
        50257: {"type": "gpt2", "original": True},
        50261: {"type": "gpt2", "original": False},
        28999: {"type": "bert-cased"},
    }
    return get_tokenizer(**vocab_sizes[vocab_size])
