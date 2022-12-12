from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import List, Union
import re
import os
import json


_UNUSED = re.compile(r"\[unused[0-9]+\]")


def load_json(path: str, encoding: str = 'utf-8') -> Union[dict, list]:
    with open(path, 'r', encoding=encoding) as r:
        return json.load(r)


def write_json(path: str, content: Union[dict, list], encoding:str = 'utf-8') -> None:
    with open(path, 'w', encoding=encoding) as w:
        json.dump(content, w)


def write_text(path: str, content: str, encoding:str = 'utf-8') -> None:
    with open(path, 'w', encoding=encoding) as w:
        w.write(content)


def add_special_tokens_to_unused(
        tokenizer: Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast],
        special_tokens: List[str],
        save_path: str = '.cache/') -> Union[PreTrainedTokenizerBase, PreTrainedTokenizerFast]:

    unused_tokens = sorted([(k, v) for k, v in tokenizer.vocab.items() if _UNUSED.match(k)], key=lambda x: x[1])
    vocab = tokenizer.vocab.copy()
    tokenizer.save_pretrained(save_path)
    vocab_json = load_json(os.path.join(save_path, 'tokenizer.json'))
    if unused_tokens:
        for spt in special_tokens:
            unu, num = unused_tokens.pop(0)
            del vocab[unu]
            del vocab_json["model"]["vocab"][unu]
            vocab[spt] = num
            vocab_json["model"]["vocab"][spt] = num

    ordered_vocab = [k for k, v in sorted(list(vocab.items()), key=lambda x: x[1])]

    write_text(os.path.join(save_path, "vocab.txt"), "\n".join(ordered_vocab))
    write_json(os.path.join(save_path, "tokenizer.json"), vocab_json)

    tokenizer = tokenizer.from_pretrained(save_path)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    tokenizer.save_pretrained(save_path)
    return tokenizer
