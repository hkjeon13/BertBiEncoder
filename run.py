from utils import add_special_tokens_to_unused
import os.path
import numpy as np
import torch
from data_collator import DataCollatorForResponseSelection
from typing import Optional, List, Union
from model import BiEncoderBertForResponseSelection
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="klue/bert-base", metadata={"help": "모델 이름 또는 경로를 설정합니다."}
    )

    model_auth_token: Optional[str] = field(
        default="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", metadata={"help": "(Optional) 모델의 인증 토큰을 입력합니다."}
    )

    projection_dimension: int = field(
        default=128, metadata={"help": ""}
    )

    label_type: str = field(
        default="binary", metadata={"help": ""}
    )

    in_batch_negative_loss: bool = field(
        default=True, metadata={"help": ""}
    )

    save_model_at_end: bool = field(
        default=True, metadata={"help": ""}
    )

    special_tokens:str = field(
        default="[SPK1],[SPK2],[SPK3],[SPK4],[SPK5]", metadata={"help": ""}
    )
@dataclass
class DataArguments:
    data_name_or_path: str = field(
        default="hknlp/aihub_ner_casual_dialogue", metadata={"help": "데이터의 이름 또는 경로를 설정합니다."}
    )

    data_auth_token: Optional[str] = field(
        default="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", metadata={"help": "(Optional) 데이터의 인증 토큰을 입력합니다."}
    )

    dialogue_column_name: str = field(
        default="dialogue", metadata={"help":""}
    )

    speaker_column_name: str = field(
        default="speaker", metadata={"help": ""}
    )

    train_split: str = field(
        default="train", metadata={"help": ""}
    )

    train_samples: Optional[int] = field(
        default=None, metadata={"help": ""}
    )

    eval_split: str = field(
        default="validation", metadata={"help": ""}
    )

    eval_samples: Optional[int] = field(
        default=None, metadata={"help": ""}
    )

    min_num_turns: int = field(
        default=3, metadata={"help": ""}
    )

    max_num_turns: int = field(
        default=10, metadata={"help": ""}
    )

    stride: int = field(
        default=1, metadata={"help": ""}
    )


class TrainParams(TrainingArguments):
    output_dir: str = field(
        default="runs/", metadata={"help": ""}
    )


def preprocess_text(text: Union[List[str], str]) -> str:
    if isinstance(text, list):
        return [preprocess_text(t) for t in text]
    text = text.strip()
    return text


def get_dialogue_response(dialogues: List[List[str]], num_turns: int, stride: int, speaker_ids: Optional[List[int]]=None, speaker_token_format="[SPK{}]"):
    output_dialogues, output_responses = [], []
    for di, dialogue in enumerate(dialogues):
        ids = [(i, i+num_turns) for i in range(0, len(dialogue), stride)]
        target_speakers = [speaker_token_format.format(s) for s in speaker_ids[di]] \
            if speaker_ids is not None else ["" for _ in range(len(ids))]

        for s, e in ids:
            targets = dialogue[s:e]
            target_spks = target_speakers[s:e]
            if len(targets) < num_turns:
                continue
            output_responses.append("".join([target_spks[-1], targets[-1]]).strip())
            output_dialogues.append([" ".join(sp_dials).strip() for sp_dials in zip(target_spks[:-1], targets[:-1])])
    return output_dialogues, output_responses


_HIT_AT_K = [1, 3, 5]

def _extract_speaker(data):
    return [d['id'] for d in data]

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainParams))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer = add_special_tokens_to_unused(tokenizer=tokenizer, special_tokens=model_args.special_tokens.split(","))

    if os.path.isdir(model_args.model_name_or_path):
        model = BiEncoderBertForResponseSelection.from_pretrained(
            model_args.model_name_or_path,
            projection_size=model_args.projection_dimension,
            label_type=model_args.label_type
        )
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = BiEncoderBertForResponseSelection(
            config,
            projection_size=model_args.projection_dimension,
            label_type=model_args.label_type,
            in_batch_negative_loss=model_args.in_batch_negative_loss
        )
        model.dialogue_bert = model.dialogue_bert.from_pretrained(model_args.model_name_or_path)
        model.response_bert = model.response_bert.from_pretrained(model_args.model_name_or_path)

    dataset = load_dataset(
        data_args.data_name_or_path,
        use_auth_token=data_args.data_auth_token
    )

    if data_args.train_samples is not None:
        length = min(data_args.train_samples, len(dataset[data_args.train_split]))
        dataset[data_args.train_split] = dataset[data_args.train_split].select(range(length))

    if data_args.eval_samples is not None:
        length = min(data_args.eval_samples, len(dataset[data_args.eval_split]))
        dataset[data_args.eval_split] = dataset[data_args.eval_split].select(range(length))

    def process_function(examples):
        examples[data_args.dialogue_column_name] = [
            preprocess_text(text)
            for text in examples[data_args.dialogue_column_name]
        ]

        speaker_ids = None
        if data_args.speaker_column_name in examples:
            speaker_ids = examples[data_args.speaker_column_name]
            speaker_ids = [_extract_speaker(sp) for sp in speaker_ids]
            new_speaker_ids = []
            for sp_ids in speaker_ids:
                unique_map = {s: (i+1) for i, s in enumerate(sorted(set(sp_ids)))}
                new_speaker_ids.append([unique_map[sp] for sp in sp_ids])
            speaker_ids = new_speaker_ids

        dialogues, responses = [], []
        for i in range(data_args.min_num_turns, data_args.max_num_turns):
            new_dialogues, new_responses = get_dialogue_response(
                dialogues=examples[data_args.dialogue_column_name],
                num_turns=i,
                stride=data_args.stride,
                speaker_ids=speaker_ids,
                speaker_token_format="[SPK{}]"
            )
            dialogues += new_dialogues
            responses += new_responses

        return {"dialogue": dialogues, "response": responses}

    dataset = dataset.map(process_function, batched=True, remove_columns=dataset[data_args.train_split].column_names)

    def example_function(examples):
        tokenized_inputs = tokenizer(
            [[tokenizer.sep_token] + utter for utter in examples['dialogue']],
            is_split_into_words=True,
            truncation=True,
            padding=True,
        )

        tokenized_inputs = {f"dialogue_{k}": v for k, v in tokenized_inputs.items()}

        tokenized_labels = tokenizer(
            examples['response'],
            truncation=True,
            padding=True,
        )

        tokenized_inputs.update(
            {f"response_{k}": v for k, v in tokenized_labels.items()}
        )

        return tokenized_inputs

    dataset = dataset.map(
        example_function,
        batched=True,
        remove_columns=dataset[data_args.train_split].column_names
    )

    collator = DataCollatorForResponseSelection(tokenizer=tokenizer)

    def compute_metrics(p):
        (preds, _, _), labels = p
        if model_args.in_batch_negative_loss:
            labels = np.argmax(labels, axis=-1)

        result = {}
        for h in _HIT_AT_K:
            candidates = np.argsort(preds, axis=-1)[:, :h]
            result[f"hit@{h}"] = sum(l in c for l, c in zip(labels, candidates))/len(labels)

        return result

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.eval_split],
    )

    if training_args.do_train:
        trainer.train()

    elif training_args.do_eval:
        result = trainer.evaluate()
        print("\n***** Evaluation Result *****")
        for k, v in result.items():
            print(f"{k}: {v}")

    if model_args.save_model_at_end:
        save_path = os.path.join(training_args.output_dir, "final")
        trainer.save_model(save_path)


if __name__ == "__main__":
    main()
