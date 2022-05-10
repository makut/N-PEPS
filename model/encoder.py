import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import params
import math

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.selu(self.linear(x))


class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, input_size, output_size):
        super(DenseBlock, self).__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_layers - 1):
            modules.append(DenseLayer(growth_rate * i + input_size, growth_rate))
        modules.append(DenseLayer(growth_rate * (num_layers - 1) + input_size, output_size))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(torch.cat(inputs, dim=-1))
            inputs.append(output)
        return inputs[-1]


class DenseEncoder(nn.Module):
    def __init__(self):
        super(DenseEncoder, self).__init__()

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.dense = DenseBlock(10, params.dense_growth_size, params.var_encoder_size * params.state_len,
                                params.dense_output_size)

    def forward(self, x):
        #print("x1:", x.shape)
        x, num_batches, num_examples = self.embed_state(x)
        #print("x2:", x.shape)
        x = F.selu(self.var_encoder(x))
        #print("x3:", x.shape)
        x = x.view(num_batches, num_examples, -1)
        #print("x4:",x.shape)
        x = self.dense(x)
        #print("x5:", x.shape)
        x = x.mean(dim=1)
        #print("x6:", x.shape)
        return x.view(num_batches, -1)

    def embed_state(self, x):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        #assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]
        num_examples = x.size()[1]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, num_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches, num_examples


class PositionalEncoding(nn.Module):
  def __init__(self, d_hid, n_position=200):
    super(PositionalEncoding, self).__init__()

    # Not a parameter
    self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

  def _get_sinusoid_encoding_table(self, n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
      return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

  def forward(self, x):
    return x + self.pos_table[:, :x.size(0)].clone().detach().transpose(0, 1)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(LearnablePositionalEncoding, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(1, n_position, d_hid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embeddings[:, :x.size(0)].transpose(0, 1)


# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.transformer_size)
        transformer_layer = nn.TransformerEncoderLayer(params.transformer_size, 8, 128)
        self.pos_encoding = LearnablePositionalEncoding(params.transformer_size)
        self.transformer = nn.TransformerEncoder(transformer_layer, 2)
        # transformer_layer = nn.TransformerEncoderLayer(params.transformer_size, 8, 256)
        # self.pos_encoding = PositionalEncoding(params.transformer_size)
        # self.transformer = nn.TransformerEncoder(transformer_layer, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, num_batches, num_examples = self.embed_state(x)
        x = self.var_encoder(x)
        x = x.permute(2, 0, 1, 3).view(-1, num_batches * num_examples, params.transformer_size)
        x = self.transformer(self.pos_encoding(x))
        x = x[0]  # CLS token
        return x.view(num_batches, num_examples, params.transformer_size).mean(dim=1)

    def embed_state(self, x):
        cls = np.tile(np.array([1, 1] + [0] * (x.shape[-1] - 2)), (x.shape[0], x.shape[1], 1, 1))
        x = torch.cat((torch.tensor(cls, dtype=torch.long, device=x.device), x), dim=2)
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        #assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len + 1, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]
        num_examples = x.size()[1]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, num_examples, params.state_len + 1, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches, num_examples


class TransformerEncoderBothWise(nn.Module):
    def __init__(self, n_layers=2, n_head=4, dim=128):
        super(TransformerEncoderBothWise, self).__init__()
        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.transformer_size)
        transformer_layer = nn.TransformerEncoderLayer(params.transformer_size, n_head, dim)
        self.pos_encoding = LearnablePositionalEncoding(params.transformer_size)
        self.transformer = nn.TransformerEncoder(transformer_layer, n_layers)
        # transformer_layer = nn.TransformerEncoderLayer(params.transformer_size, 8, 256)
        # self.pos_encoding = PositionalEncoding(params.transformer_size)
        # self.transformer = nn.TransformerEncoder(transformer_layer, 8)

        self.projector = nn.Linear(params.transformer_size * params.state_len, params.dense_output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, num_batches, num_examples = self.embed_state(x)
        x = F.selu(self.var_encoder(x))
        x = x.permute(1, 0, 2)
        x = self.transformer(self.pos_encoding(x))
        x = x.permute(1, 0, 2)
        assert x.shape == (num_batches, num_examples * params.state_len, params.transformer_size)
        x = x.reshape(num_batches, num_examples, -1)
        x = F.selu(self.projector(x)).mean(dim=1)

        # x = x[0]  # CLS token
        return x

    def embed_state(self, x):
        # x = torch.cat((torch.tensor(cls, dtype=torch.long, device=x.device), x), dim=2)
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        #assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]
        num_examples = x.size()[1]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, num_examples, params.state_len, -1)
        types = types.contiguous().float()
        concated = torch.cat((embedded_values, types), dim=-1)
        concated = concated.view(num_batches, num_examples * params.state_len, -1)
        # cls = np.tile(np.array([0] * (concated.shape[-1] - 2) + [1, 1]), (concated.shape[0], 1, 1))
        # print(concated.shape, cls.shape, embedded_values.shape)
        # concated = torch.cat((torch.tensor(cls, dtype=x.dtype, device=x.device), concated), dim=1)
        assert concated.size()[0] == num_batches
        assert concated.size()[1] == params.state_len * num_examples
        assert concated.size()[2] == embedded_values.size()[-1] + types.size()[-1]
        return concated, num_batches, num_examples


class RomaDenseEncoder(nn.Module):
    def __init__(self):
        super(RomaDenseEncoder, self).__init__()

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.transformer_layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(params.var_encoder_size, 8, 1024, activation=F.selu, batch_first=True) for _ in
             range(5)])
        self.encoder = nn.Linear(params.var_encoder_size * params.state_len, params.dense_output_size)

    def forward(self, x):
        n_examples = x.shape[1]
        x, num_batches = self.embed_state(x, n_examples)
        x = F.selu(self.var_encoder(x)).view(num_batches * n_examples, params.state_len, -1)

        for layer in self.transformer_layers:
            x = layer(x)

        x = x.view(num_batches, n_examples, -1)
        x = F.selu(self.encoder(x))

        return x.mean(dim=1)

    def embed_state(self, x, n_examples):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        assert values.size()[1] == n_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, n_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches


"""    
class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.selu(self.linear(x))


class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, input_size, output_size):
        super(DenseBlock, self).__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_layers - 1):
            modules.append(DenseLayer(growth_rate * i + input_size, growth_rate))
        modules.append(DenseLayer(growth_rate * (num_layers - 1) + input_size, output_size))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(torch.cat(inputs, dim=-1))
            inputs.append(output)
        return inputs[-1]


class DenseEncoder(nn.Module):
    def __init__(self):
        super(DenseEncoder, self).__init__()

        self.embedding = nn.Embedding(params.integer_range + 1, params.embedding_size)
        self.var_encoder = nn.Linear(params.max_list_len * params.embedding_size + params.type_vector_len,
                                     params.var_encoder_size)
        self.dense = DenseBlock(10, params.dense_growth_size, params.var_encoder_size * params.state_len,
                                params.dense_output_size)

    def forward(self, x):
        x, num_batches = self.embed_state(x)
        x = F.selu(self.var_encoder(x))
        x = x.view(num_batches, params.num_examples, -1)
        x = self.dense(x)
        x = x.mean(dim=1)
        return x.view(num_batches, -1)

    def embed_state(self, x):
        types = x[:, :, :, :params.type_vector_len]
        values = x[:, :, :, params.type_vector_len:]

        assert values.size()[1] == params.num_examples, "Invalid num of examples received!"
        assert values.size()[2] == params.state_len, "Example with invalid length received!"
        assert values.size()[3] == params.max_list_len, "Example with invalid length received!"

        num_batches = x.size()[0]

        embedded_values = self.embedding(values.contiguous().view(num_batches, -1))
        embedded_values = embedded_values.view(num_batches, params.num_examples, params.state_len, -1)
        types = types.contiguous().float()
        return torch.cat((embedded_values, types), dim=-1), num_batches
"""

