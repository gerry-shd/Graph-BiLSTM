import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTM_Attention(nn.Module):
    def __init__(self, config):
        super(BiLSTM_Attention, self).__init__()
        self.config = config
        self.input_size = self.config.embedding_size * self.config.channels
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.is_bidirectional = self.config.is_bidirectional
        self.num_directions = 1
        if self.is_bidirectional:
            self.num_directions = 2

        self.dropout_p = config.dropout

        self.bilstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                              num_layers=self.num_layers, bidirectional=self.is_bidirectional)

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.relu = nn.LeakyReLU()
        self.W = nn.Parameter(torch.Tensor(self.hidden_size * self.num_layers, self.hidden_size * self.num_layers))
        self.U = nn.Parameter(torch.Tensor(self.hidden_size * self.num_layers, 1))

        self.linear = nn.Linear(config.hidden_size * self.num_layers, config.num_class)

        nn.init.uniform_(self.W, -0.1, 0.1)
        nn.init.uniform_(self.U, -0.1, 0.1)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [sentence_length, batch_size, embedding_size]

        batch_size = x.size(1)
        h_0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(self.config.device)
        c_0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(self.config.device)

        # out[seq_len, batch_size, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        # h_n, c_n [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.bilstm(x, (h_0, c_0))
        out = self.dropout(out)  # x [batch_size,max_node_len,embedding_size]
        x = out  # [num_layers*num_directions, batch_size, hidden_size]
        x = x.permute(1, 0, 2)  # [batch_size, num_layers*num_directions, hidden_size]
        # tanh attention
        score = torch.tanh(torch.matmul(x, self.W))  # [batch_size, max_node_len,hidden_size*2]
        attention = F.softmax(torch.matmul(score, self.U), dim=1)
        score_x = x * attention
        feat = torch.sum(score_x, dim=1)  # [batch_size,hidden_size*2]
        return self.linear(feat)
