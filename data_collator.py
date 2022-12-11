from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Union, Optional, List, Dict, Any
import torch

@dataclass
class DataCollatorForResponseSelection:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        dialogue_features, response_features, others = [], [], []
        for f in features:
            dialogue_feature, response_feature, other = {}, {}, {}
            for k, v in f.items():
                if k.startswith("dialogue_"):
                    dialogue_feature[k.replace("dialogue_", "")] = v
                elif k.startswith("response_"):
                    response_feature[k.replace("response_", "")] = v
                else:
                    other[k] = v
            dialogue_features.append(dialogue_feature)
            response_features.append(response_feature)
            others.append(other)


        dialogue_batch = self.tokenizer.pad(
            dialogue_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        response_batch = self.tokenizer.pad(
            response_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        others_batch = {}
        if others[0]:
            others_batch.update(self.tokenizer.pad(
                others,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            ))

        batch = dict(
            {f"dialogue_{k}": v for k, v in dialogue_batch.items()},
            **{f"response_{k}": v for k, v in response_batch.items()},
            **{k: v for k, v in other.items()}
        )

        batch["label"] = [1. for _ in range(len(dialogue_features))]
        if self.return_tensors == "pt":
            batch['label'] = torch.tensor(batch['label'])

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]

        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch