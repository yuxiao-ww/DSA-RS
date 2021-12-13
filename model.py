import torch.nn as nn
from transformers import (
    BertModel,
    BertForSequenceClassification,
)


class sentenceLevelBert(nn.Module):
    def __init__(self, FLAGS, config):
        super(sentenceLevelBert, self).__init__()
        # BERT encode sentences
        self.bert = BertModel.from_pretrained(FLAGS.model_version, config=config)
        self.bert.to(FLAGS.device)

    def forward(self, data):
        out = self.bert(data)
        return out['pooler_output']


class documentLevelBert(nn.Module):
    def __init__(self, FLAGS, config, num_embeddings=None, up_vocab=None):
        super(documentLevelBert, self).__init__()
        self.num_labels = len(FLAGS.classes)
        self.model = BertForSequenceClassification.from_pretrained(
            FLAGS.model_version,
            config=config
        )
        # print(bert_model)
        self.FLAGS = FLAGS
        self.model.to(FLAGS.device)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.model(input_ids=input_ids.long(),
                       token_type_ids=token_type_ids.long(),
                       attention_mask=attention_mask.long(),
                       labels=labels.long(),
                       )  # 输出loss 和 每个分类对应的输出，softmax后才是预测是对应分类的概率
        x_loss = x[0]
        # print(x_loss)
        x_logits = x[1].squeeze()
        # print(x_logits.shape)
        x_logits = self.softmax(x_logits)
        return x_loss, x_logits
