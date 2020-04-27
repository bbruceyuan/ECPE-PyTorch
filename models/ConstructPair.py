from .BaseModel import BaseModel
from .encoders.BiLSTMAttentionEncoder import BiLSTMAttentionEncoder
import torch
import torch.nn as nn


class ConstructPair(BaseModel):
    def __init__(self, args, word_embedding_lookup_table, position_lookup_table):
        super(ConstructPair, self).__init__()
        self.args = args
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding_lookup_table, padding_idx=0)
        self.position_embedding = nn.Embedding.from_pretrained(position_lookup_table, padding_idx=0)

        self.sentence_encoder = BiLSTMAttentionEncoder(args, word_embedding_lookup_table)

        self.embedding_dropout = nn.Dropout(self.args.embedding_drop)
        self.softmax_dropout = nn.Dropout(self.args.softmax_drop)

        self.classifier = nn.Linear(2 * self.args.lstm_hidden_dim + self.args.embedding_dim_pos, 2)

    def forward(self, inputs, distances, lengths=None):
        x = self.word_embedding(inputs)
        x = x.view(-1, self.args.max_clause_len, self.args.embedding_dim)
        x = self.embedding_dropout(x)
        x = self.sentence_encoder(x)
        # x shape is (batch, hidden_dim)
        x = x.view(-1, 2 * self.args.lstm_hidden_dim)
        distance_embedding = self.position_embedding(distances)
        x = torch.cat((x, distance_embedding), dim=1)
        # 这里可以加上 softmax 之前的 dropout ，可能会有用
        x = self.classifier(x)
        # x shape is (batch, 2)
        return x
