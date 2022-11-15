import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

class ConvolutionLayer(nn.Module):
    def __init__(self, input_size, channels, dilation, dropout_probability=0.1):
        super(ConvolutionLayer, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.dilation = dilation
        self.dropout_probability = dropout_probability
    
    def forward(self, inputs):
        base = nn.Sequential(
            nn.Dropout2d(self.dropout_probability),
            nn.Conv2d(self.input_size, self.channels, kernel_size=1),
            nn.GELU(),
        )
        convs = []
        for d in self.dilation:
            convs.append(nn.Conv2d(self.channels, self.channels, kernel_size=3, groups=self.channels, dilation=d, padding=d))
        convs = nn.ModuleList(convs)

        inputs = base(inputs.permute(0, 3, 1, 2))
        outputs = []
        for conv in self.convs:
            inputs = F.gelu(conv(inputs))
            outputs.append(inputs)
        outputs = torch.cat(outputs, dim=1).permute(0, 2, 3, 1)
        return outputs

class PredictorBiaffine(nn.Module):
    def __init__(self, in_features_size, out_features_size=1, bias_x=True, bias_y=True):
        super(PredictorBiaffine, self).__init__()
        self.in_features_size = in_features_size
        self.out_features_size = out_features_size
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((out_features_size, in_features_size + int(bias_x), in_features_size + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, h_i, h_j):
        if self.bias_x:
            h_i = torch.cat((h_i, torch.ones_like(h_i[..., :1])), -1)
        if self.bias_y:
            h_j = torch.cat((h_j, torch.ones_like(h_j[..., :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', h_i, self.weight, h_j).permute(0, 2, 3, 1)
        return s

class PredictorMLP(nn.Module):
    def __init__(self, in_features_size, out_features_size, dropout_probability=0):
        super().__init__()
        self.in_features_size = in_features_size
        self.out_features_size = out_features_size
        self.dropout_probability = dropout_probability

    def forward(self, inputs):
        linear = nn.Linear(self.in_features_size, self.out_features_size)
        gelu = nn.GELU()
        dropout = nn.Dropout(self.dropout_probability)
        outputs = gelu(linear(dropout(inputs)))
        return outputs

class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout_probability=0):
        super().__init__()
        self.cls_num = cls_num
        self.hid_size = hid_size
        self.biaffine_size = biaffine_size
        self.channels = channels
        self.ffnn_hid_size = ffnn_hid_size
        self.dropout_probability = dropout_probability

    def forward(self, h_i, h_j, q_ij):
        mlp1 = PredictorMLP(in_features_size=self.hid_size, out_features_size=self.biaffine_size, dropout_probability=self.dropout_probability)
        mlp2 = PredictorMLP(in_features_size=self.hid_size, out_features_size=self.biaffine_size, dropout_probability=self.dropout_probability)
        biaffine = PredictorBiaffine(in_features_size=self.biaffine_size, out_features_size=self.cls_num, bias_x=True, bias_y=True)
        mlp_rel = PredictorMLP(self.channels, self.ffnn_hid_size, dropout_probability=self.dropout_probability)
        linear = nn.Linear(self.ffnn_hid_size, self.cls_num)
        dropout = nn.Dropout(self.dropout_probability)
        y1_ij = biaffine(dropout(mlp1(h_i)), dropout(mlp2(h_j)))
        y2_ij = linear(dropout(mlp_rel(q_ij)))
        return y1_ij+y2_ij

class LayerNormalization(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver'):
        super(LayerNormalization, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
class W2NERmodel(nn.Module):
    def __init__(self, config):
        super(W2NERmodel, self).__init__()
        self.config = config

    def forward(self, bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length):
        bert_name = self.config.bert_name
        use_bert_last_4_layers = self.config.use_bert_last_4_layers
        lstm_hid_size = self.config.lstm_hid_size
        conv_hid_size = self.config.conv_hid_size
        lstm_input_size = 0
        dilation = self.config.dilation
        biaffine_size = self.config.biaffine_size
        out_dropout = self.config.out_dropout
        conv_dropout = self.config.conv_dropout
        ffnn_hid_size = self.config.ffnn_hid_size
        lstm_input_size += self.config.bert_hid_size
        dist_emb_size = self.config.dist_emb_size
        type_emb_size = self.config.type_emb_size
        label_num = self.config.label_num
        emb_dropout = self.config.emb_dropout

        dis_embs = nn.Embedding(20, dist_emb_size)
        reg_embs = nn.Embedding(3, type_emb_size)
        bert = AutoModel.from_pretrained(bert_name, cache_dir="./cache/", output_hidden_states=True)
        encoder = nn.LSTM(lstm_input_size, lstm_hid_size // 2, num_layers=1, batch_first=True, bidirectional=True)
        conv_input_size = lstm_hid_size + dist_emb_size + type_emb_size
        convLayer = ConvolutionLayer(conv_input_size, conv_hid_size, dilation, conv_dropout)
        dropout = nn.Dropout(emb_dropout)
        predictor = CoPredictor(label_num, lstm_hid_size, biaffine_size,
                                     conv_hid_size * len(dilation), ffnn_hid_size, out_dropout)

        cln = LayerNormalization(lstm_hid_size, lstm_hid_size, conditional=True)

        bert_embs = bert(input_ids=bert_inputs, attention_mask=bert_inputs.ne(0).float())
        if use_bert_last_4_layers:
            bert_embs = torch.stack(bert_embs[2][-4:], dim=-1).mean(-1)
        else:
            bert_embs = bert_embs[0]

        length = pieces2word.size(1)

        min_value = torch.min(bert_embs).item()

        _bert_embs = bert_embs.unsqueeze(1).expand(-1, length, -1, -1)
        _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
        word_reps, _ = torch.max(_bert_embs, dim=2)

        word_reps = dropout(word_reps)
        packed_embs = pack_padded_sequence(word_reps, sent_length.cpu(), batch_first=True, enforce_sorted=False)
        packed_outs, (hidden, _) = encoder(packed_embs)
        word_reps, _ = pad_packed_sequence(packed_outs, batch_first=True, total_length=sent_length.max())

        cln = cln(word_reps.unsqueeze(2), word_reps)

        dis_emb = dis_embs(dist_inputs)
        tril_mask = torch.tril(grid_mask2d.clone().long())
        reg_inputs = tril_mask + grid_mask2d.clone().long()
        reg_emb = reg_embs(reg_inputs)

        conv_inputs = torch.cat([dis_emb, reg_emb, cln], dim=-1)
        conv_inputs = torch.masked_fill(conv_inputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        conv_outputs = convLayer(conv_inputs)
        conv_outputs = torch.masked_fill(conv_outputs, grid_mask2d.eq(0).unsqueeze(-1), 0.0)
        outputs = predictor(word_reps, word_reps, conv_outputs)

        return outputs
