from dataclasses import dataclass
from typing import Optional, Union, Tuple
import os
import torch
from torch import nn
from transformers import BertModel, PreTrainedModel
from torch.nn import BCELoss, CosineSimilarity, ReLU, CosineEmbeddingLoss, CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertConfig,
    load_tf_weights_in_bert,
    BertPreTrainedModel
)


@dataclass
class ResponseSelectionOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    """

    loss: Optional[torch.FloatTensor] = None
    similarity: torch.FloatTensor = None
    dialogue_embeddings: Optional[Tuple[torch.FloatTensor]] = None
    response_embeddings: Optional[Tuple[torch.FloatTensor]] = None


class BiEncoderBertForResponseSelection(BertPreTrainedModel):
    def __init__(
            self,
            config,
            projection_size: int = 128,
            label_type: str = "binary",
            in_batch_negative_loss: bool = True
    ) -> None:
        super().__init__(config)
        self.config = config
        self.config.label_type = label_type
        self.config.in_batch_negative_loss = in_batch_negative_loss
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        print(config)
        self.dialogue_bert = BertModel(config)
        self.dialogue_dropout = nn.Dropout(classifier_dropout)
        self.dialogue_projection = nn.Linear(config.hidden_size, projection_size)

        self.response_bert = BertModel(config)
        self.response_dropout = nn.Dropout(classifier_dropout)
        self.response_projection = nn.Linear(config.hidden_size, projection_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        dialogue_input_ids: Optional[torch.Tensor] = None,
        dialogue_attention_mask: Optional[torch.Tensor] = None,
        dialogue_token_type_ids: Optional[torch.Tensor] = None,
        dialogue_position_ids: Optional[torch.Tensor] = None,
        dialogue_head_mask: Optional[torch.Tensor] = None,
        dialogue_inputs_embeds: Optional[torch.Tensor] = None,
        dialogue_output_attentions: Optional[bool] = None,
        dialogue_output_hidden_states: Optional[bool] = None,
        response_input_ids: Optional[torch.Tensor] = None,
        response_attention_mask: Optional[torch.Tensor] = None,
        response_token_type_ids: Optional[torch.Tensor] = None,
        response_position_ids: Optional[torch.Tensor] = None,
        response_head_mask: Optional[torch.Tensor] = None,
        response_inputs_embeds: Optional[torch.Tensor] = None,
        response_output_attentions: Optional[bool] = None,
        response_output_hidden_states: Optional[bool] = None,

        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], ResponseSelectionOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dialogue_outputs = self.dialogue_bert(
            dialogue_input_ids,
            attention_mask=dialogue_attention_mask,
            token_type_ids=dialogue_token_type_ids,
            position_ids=dialogue_position_ids,
            head_mask=dialogue_head_mask,
            inputs_embeds=dialogue_inputs_embeds,
            output_attentions=dialogue_output_attentions,
            output_hidden_states=dialogue_output_hidden_states,
            return_dict=return_dict,
        )

        response_outputs = self.response_bert(
            response_input_ids,
            attention_mask=response_attention_mask,
            token_type_ids=response_token_type_ids,
            position_ids=response_position_ids,
            head_mask=response_head_mask,
            inputs_embeds=response_inputs_embeds,
            output_attentions=response_output_attentions,
            output_hidden_states=response_output_hidden_states,
            return_dict=return_dict,
        )

        dialogue_pooled_output = dialogue_outputs[1]
        dialogue_pooled_output = self.dialogue_dropout(dialogue_pooled_output)
        dialogue_logits = self.dialogue_projection(dialogue_pooled_output)

        response_pooled_output = response_outputs[1]
        response_pooled_output = self.response_dropout(response_pooled_output)
        response_logits = self.response_projection(response_pooled_output)

        similarity = torch.matmul(dialogue_logits, response_logits.t())

        loss = None
        if labels is not None:
            if self.config.label_type == "binary":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(similarity, labels)
            elif self.config.label_type == "polarity":
                loss_fct = CosineEmbeddingLoss()
                loss = loss_fct(dialogue_logits, response_logits, labels)
            else:
                raise NotImplementedError(
                    f"The '{self.label_type}' loss function is not implemented yet!"
                )
        if not return_dict:
            output = (similarity, dialogue_logits, response_logits)
            return ((loss,) + output) if loss is not None else output

        return ResponseSelectionOutput(
            loss=loss,
            similarity=similarity,
            dialogue_embeddings=dialogue_logits,
            response_embeddings=response_logits,
        )


if __name__ == "__main__":
    model = BiEncoderBertForResponseSelection.from_pretrained("klue/bert-base", cache_dir=".cache")
