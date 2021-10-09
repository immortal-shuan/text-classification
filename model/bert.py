import torch.nn as nn
from transformers import BertModel


class model_classification(nn.Module):
    def __init__(self, args):
        super(model_classification, self).__init__()

        self.bert_model = BertModel.from_pretrained('F:/pretrain_model/bert_wwm')
        for param in self.bert_model.parameters():
            param.requires_grad = True

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.bert_dim, args.num_class)

    def forward(self, input_id, attention_mask):

        word_vec = self.bert_model(
            input_ids=input_id, attention_mask=attention_mask
        )
        word_vec = word_vec.pooler_output
        word_vec = self.dropout(word_vec)
        out = self.fc(word_vec)
        return out


