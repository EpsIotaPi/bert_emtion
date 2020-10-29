import torch
from torch import nn

from transformers import BertModel, BertConfig


class HanAttention(nn.Module):

    def __init__(self, hidden_dim):
        super(HanAttention, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.m = nn.Softmax(dim=1)

    def forward(self, inputs):
        u = torch.tanh(self.linear(inputs))
        attention = self.linear_v(u)
        # attention = attention.masked_fill(mask == 0, -1e10)
        alphas = self.m(attention)
        outputs = torch.mul(alphas, inputs)
        outputs = torch.sum(outputs, dim=1)
        return outputs, alphas




class BertWwmModel(nn.Module):
    name = 'BertWwmModel_'

    def __init__(self, params, model='', bidirectional=False):
        super(BertWwmModel, self).__init__()
        self.model = model
        self.bidrectional = bidirectional
        if bidirectional and model != '':
            self.name += "bi-"
        self.name += model

        self.dropout = nn.Dropout(params.dropout_rate)
        self.attention = HanAttention(params.hidden_size)
        self.lstm = nn.LSTM(params.embed_dim,
                            params.hidden_size,
                            num_layers=params.rnn_layers,
                            bidirectional=bidirectional,
                            dropout=params.dropout_rate,
                            batch_first=True)
        self.gru = nn.GRU(params.embed_dim,
                          params.hidden_size,
                          num_layers=params.rnn_layers,
                          bidirectional=bidirectional,
                          dropout=params.dropout_rate,
                          batch_first=True)
        self.combined_linear = nn.Sequential(nn.Linear(2 * params.hidden_size,params.hidden_size),
                                             nn.ReLU())
        config = BertConfig.from_pretrained(params.bert_config_path)
        self.bert_model = BertModel.from_pretrained(params.bert_path, config=config)
        self.fc = nn.Sequential(self.dropout,
                                nn.Linear(params.hidden_size, params.hidden_size),
                                nn.ReLU(inplace=True),
                                self.dropout,
                                nn.Linear(params.hidden_size, params.n_classes))

    def forward(self, sentences):
        # text
        input_ids, attention_mask, token_type_ids = sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"]
        # batch_size, length. hidden_dim (5,512,768)
        text_embedding, _ = self.bert_model(input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)
        text_model = text_embedding
        if self.model == 'lstm':
            text_model, _ = self.lstm(text_model)
        elif self.model == 'gru':
            text_model, _ = self.gru(text_model)
        if self.bidrectional:
            text_model = self.combined_linear(text_model)

        # text_attention shape  batch , hidden_dim
        text_attention, _ = self.attention(text_model)
        prob = self.fc(text_attention)
        return prob


# + cls 看下效果 VS attention
# + 加上BiGRU、BiLSTM、CNN  VS  max(cls , attention)


class RoBertaModel(nn.Module):
    name = 'RoBertaModel_'

    def __init__(self, params, model='', bidirectional=False):
        super(RoBertaModel, self).__init__()
        self.model = model
        self.bidirectional = bidirectional
        if bidirectional and model != '':
            self.name += "bi-"
        self.name += model
        self.dropout = nn.Dropout(params.dropout_rate)
        self.attention = HanAttention(params.hidden_size)
        self.lstm = nn.LSTM(params.embed_dim,
                            params.hidden_size,
                            num_layers=params.rnn_layers,
                            bidirectional=bidirectional,
                            dropout=params.dropout_rate,
                            batch_first=True)
        self.gru = nn.GRU(params.embed_dim,
                          params.hidden_size,
                          num_layers=params.rnn_layers,
                          bidirectional=bidirectional,
                          dropout=params.dropout_rate,
                          batch_first=True)
        # self.cnn = nn.Conv3d(params.embed_dim,
        #                      params.hidden_size,
        #                      params.hidden_size)
        self.combined_linear = nn.Sequential(nn.Linear(2 * params.hidden_size, params.hidden_size),
                                             nn.ReLU())
        config = BertConfig.from_pretrained(params.roberta_config_path)
        self.bert_model = BertModel.from_pretrained(params.roberta_path, config=config)
        self.fc = nn.Sequential(self.dropout,
                                nn.Linear(params.hidden_size, params.hidden_size),
                                nn.ReLU(inplace=True),
                                self.dropout,
                                nn.Linear(params.hidden_size, params.n_classes))

    def forward(self, sentences):
        # text
        input_ids, attention_mask, token_type_ids = sentences["input_ids"], sentences["attention_mask"], sentences["token_type_ids"]
        text_embedding, _ = self.bert_model(input_ids,
                                            attention_mask=attention_mask,
                                            token_type_ids=token_type_ids)
        text_model = text_embedding
        if self.model == 'lstm':
            text_model, _ = self.lstm(text_model)
        elif self.model == 'gru':
            text_model, _ = self.gru(text_model)
        if self.bidirectional:
            text_model = self.combined_linear(text_model)

        text_attention, _ = self.attention(text_model)
        prob = self.fc(text_attention)
        return prob