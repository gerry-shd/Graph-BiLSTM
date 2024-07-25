import torch
import torch.nn as nn
# from graph_bilstm import config as config


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        self.input_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.is_bidirectional = config.is_bidirectional
        self.lstm = nn.LSTM(config.embedding_size * config.channels, config.hidden_size, num_layers=config.num_layers,
                            bidirectional=config.is_bidirectional)

        self.linear = nn.Linear(config.num_layers * 2 * config.hidden_size, config.num_class)
        self.act_func = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # [sentence_length, batch_size, embedding_size]

        batch_size = x.size(1)
        h_0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(self.config.device)
        c_0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(self.config.device)

        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        x = h_n  # [num_layers*num_directions, batch_size, hidden_size]
        x = x.permute(1, 0, 2)  # [batch_size, num_layers*num_directions, hidden_size]
        x = x.contiguous().view(batch_size,
                                self.num_layers * 2 * self.hidden_size)  # [batch_size, num_layers*num_directions*hidden_size]
        x = self.linear(x)
        x = self.act_func(x)
        # print(x.shape)
        return x
