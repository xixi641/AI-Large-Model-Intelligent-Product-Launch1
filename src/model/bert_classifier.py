from torch import nn
from configuration import config
from transformers import AutoModel


class ProductClassifier(nn.Module):

    def __init__(self,freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')
        self.liner = nn.Linear(self.bert.config.hidden_size, config.NUMBER_CLASSES)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = not freeze_bert


    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        cls_output = last_hidden[:, 0, :]
        return self.liner(cls_output)