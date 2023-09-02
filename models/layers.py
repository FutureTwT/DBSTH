import math
import torch
import torch.nn as nn
import pickle as pkl

from collections import OrderedDict


class ReweightSparseFusion(nn.Module):
    def __init__(self, patch_dim, concept_lens) -> None:
      super(ReweightSparseFusion, self).__init__()
      self.att = nn.Sequential(
          nn.Linear(patch_dim, concept_lens),
          nn.Sigmoid())

    def forward(self, x):
      sparse_weight = self.att(x) # (N, L, D) -> (N, L, K) 
      output = torch.einsum('nld,nlk->nkd', x, sparse_weight)
      return output
    

# Check in 2023-06-06
def patch_masking(x: torch.Tensor, mask_ratio=0.5)->torch.Tensor:
    N, K, D = x.shape
    keep_mask = torch.zeros(N, K)
    num = int(K * (1 - mask_ratio)) # number of keeping patches

    for i in range(N):
      indices = torch.randperm(K)
      keep_mask[i, indices[:num]] = 1
    keep_mask = keep_mask.unsqueeze(-1).expand(-1, -1, D).cuda()

    x = x[keep_mask.bool()].view(N, -1, D)
    return x
    

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    Refer:
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FFN(nn.Module):
    def __init__(self, hidden_dim=[768, 2048, 512], act=nn.Tanh()):
        super(FFN, self).__init__()
        self.hidden_dim = hidden_dim

        orderedDict = OrderedDict()
        for i in range(len(hidden_dim) - 1):
            index = i + 1
            orderedDict['linear' + str(index)] = nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1])
            orderedDict['bn' + str(index)] = nn.LayerNorm(self.hidden_dim[i + 1])
            orderedDict['act' + str(index)] = act

        self.mlp = nn.Sequential(orderedDict)
        # self._initialize()

    def _initialize(self):
        nn.init.xavier_normal_(self.mlp.linear1.weight.data)
        nn.init.xavier_normal_(self.mlp.linear2.weight.data)

    def forward(self, x):
        return self.mlp(x)

