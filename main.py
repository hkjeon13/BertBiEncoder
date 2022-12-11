import os.path

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


def get_dialogue_response(dialogues: List[List[str]], num_turns: int, stride: int):
    output_dialogues, output_responses = [], []
    for dialogue in dialogues:
        ids = [(i, i+num_turns) for i in range(0, len(dialogue), stride)]
        for s, e in ids:
            targets = dialogue[s:e]
            if len(targets) < num_turns:
                continue
            output_responses.append(targets[-1])
            output_dialogues.append(targets[:-1])
    return output_dialogues, output_responses


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainParams))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

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

        dialogues, responses = [], []
        for i in range(data_args.min_num_turns, data_args.max_num_turns):
            new_dialogues, new_responses = get_dialogue_response(
                dialogues=examples[data_args.dialogue_column_name],
                num_turns=i,
                stride=data_args.stride,
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
            padding="max_length",
            max_length=256
        )

        tokenized_inputs.update(
            {f"response_{k}": v for k, v in tokenized_labels.items()}
        )

        return tokenized_inputs

    dataset = dataset.map(example_function, batched=True, remove_columns=dataset[data_args.train_split].column_names)
    collator = DataCollatorForResponseSelection(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args,
        train_dataset=dataset[data_args.train_split],
        eval_dataset=dataset[data_args.eval_split],
    )

    if training_args.do_train:
        trainer.train()
    elif training_args.do_eval:
        trainer.evaluate()

    save_path = os.path.join(training_args.output_dir, "final")
    trainer.save_model(save_path)


if __name__ == "__main__":
    main()
