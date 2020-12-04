import torch

from torch import nn


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert_model = bert
        self.dr_rate = dr_rate # dropout rate
        self.classifier = nn.Linear(hidden_size, num_classes)

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_len):
        attention_mask = torch.zeros_like(token_ids)

        for i, v in enumerate(valid_len):
            attention_mask[i][:v] = 1

        return attention_mask.float()

    def forward(self, token_ids, valid_len, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_len)

        sequence_output, pooled_output = self.bert_model(input_ids=token_ids, token_type_ids=segment_ids.long(),
                                                         attention_mask=attention_mask.float().to(token_ids.device))
        out = pooled_output

        if self.dr_rate:
            out = self.dropout(pooled_output)

        out = self.classifier(out)

        return out
