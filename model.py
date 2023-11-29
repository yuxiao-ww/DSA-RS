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


class DSC_RS(nn.Module):
    def __init__(self, FLAGS, config, num_embeddings=None, up_vocab=None):
        super(DSC_RS, self).__init__()
        self.document_bert = documentLevelBert(FLAGS, config)

        self.wide_deep = WideAndDeepModel(wide_deep_params)
        self.ncf = NCFModel(ncf_params)

        self.processing_layer = nn.Linear(wide_deep_params['output_dim'] + ncf_params['output_dim'], FLAGS.bert_input_dim)
        self.additional_classifier = nn.Linear(FLAGS.bert_input_dim + FLAGS.bert_output_dim, FLAGS.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, wide_input, deep_input, user_input,
                item_input):
        # Get Cp and Cu from the respective models
        C_p = self.wide_deep(wide_input, deep_input)
        C_u = self.ncf(user_input, item_input)

        # Process C_p and C_u before concatenation
        combined_Cp_Cu = self.processing_layer(torch.cat([C_p, C_u], dim=1))
        # Pass inputs to the document-level BERT model and obtain loss and logits
        x_loss, x_logits = self.document_bert(input_ids, attention_mask, token_type_ids, labels)
        # Combine the processed C_p and C_u with the BERT output
        combined_features = torch.cat([x_logits, combined_Cp_Cu], dim=1)
        # Pass the combined features through the additional classifier
        classifier_output = self.additional_classifier(combined_features)

        return x_loss, classifier_output


wide_deep_params = {
    'wide_dim': 100,
    'deep_dim': 64,
    'hidden_units': [128, 64],
    'output_dim': 1
}

ncf_params = {
    'num_users': 10000,
    'num_items': 1000,
    'general_dim': 8,
    'hidden_units': [64, 32, 16]
}
