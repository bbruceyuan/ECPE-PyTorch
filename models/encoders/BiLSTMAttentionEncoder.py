import torch.nn as nn
from layers.attention import WeightedAttention
import logging

logger = logging.getLogger(__name__)


class BiLSTMAttentionEncoder(nn.Module):
    def __init__(self, args, word_embedding_lookup_table=None):
        super(BiLSTMAttentionEncoder, self).__init__()
        self.args = args

        # 根据 word vec 的矩阵进行初始化
        self.embedding = nn.Embedding.from_pretrained(word_embedding_lookup_table, padding_idx=0)

        self.attention = WeightedAttention(self.args.lstm_hidden_dim, self.args.lstm_hidden_dim * 2)
        # 可替换成 gru, rnn 之类的
        self.rnn = nn.LSTM(self.args.embedding_dim,
                           self.args.lstm_hidden_dim // 2,
                           bidirectional=True,
                           batch_first=True)

    def forward(self, inputs, lengths=None):
        # 如果是按照 lengths 排序，后面还需要恢复 doc 中 clause 的真正顺序
        # inputs shape is (batch, max_clause_len)

        # 根据 lengths 进行句子 pack
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.rnn(inputs)
        # 进行 pad
        # out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]

        # out shape is (batch, clause_len, lstm_hidden_dim)
        out, _ = self.attention(out)
        # out shape is (batch, lstm_hidden_dim)
        return out
