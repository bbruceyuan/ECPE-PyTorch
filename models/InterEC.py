import torch
import torch.nn as nn
import logging
from .encoders.BiLSTMAttentionEncoder import BiLSTMAttentionEncoder
from .BaseModel import BaseModel

logger = logging.getLogger(__name__)


class InterEC(BaseModel):
    def __init__(self, args, word_embedding_lookup_table):
        super(InterEC, self).__init__()
        self.args = args
        # 输入层编码
        self.emotion_encoder = BiLSTMAttentionEncoder(args, word_embedding_lookup_table=word_embedding_lookup_table)
        self.cause_encoder = BiLSTMAttentionEncoder(args, word_embedding_lookup_table=word_embedding_lookup_table)

        if word_embedding_lookup_table is None:
            self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_dim)
        else:
            # 根据 word vec 的矩阵进行初始化
            self.embedding = nn.Embedding.from_pretrained(word_embedding_lookup_table, padding_idx=0)
        self.embedding_dropout = nn.Dropout(self.args.embedding_drop)
        self.emotion_lstm = nn.LSTM(
            self.args.lstm_hidden_dim,
            self.args.lstm_hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        self.cause_lstm = nn.LSTM(
            self.args.lstm_hidden_dim + 2,
            self.args.lstm_hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(self.args.lstm_hidden_dim, self.args.n_class)
        )
        self.cause_classifier = nn.Sequential(
            nn.Linear(self.args.lstm_hidden_dim, self.args.n_class)
        )

    def forward(self,
                inputs,
                ):
        # input x 是 (batch, max_doc_len,  max_clause_len)
        inputs = inputs.view(-1, self.args.max_clause_len)

        inputs = self.embedding(inputs)
        inputs = self.embedding_dropout(inputs)

        emotion_x = self.emotion_encoder(inputs)

        # 根据 max_doc_len 进行 reshape
        emotion_x = emotion_x.view(-1, self.args.max_doc_len, self.args.lstm_hidden_dim)
        # x shape is (new_batch, max_doc_len, hidden_dim)

        emotion_x, _ = self.emotion_lstm(emotion_x)
        # 分类是不是 emotion
        predict_emotion_logits = self.emotion_classifier(emotion_x)

        # 同样的操作需要对 cause 进行一遍
        cause_x = self.cause_encoder(inputs)
        cause_x = cause_x.view(-1, self.args.max_doc_len, self.args.lstm_hidden_dim)

        # 与 inter ce 中的 操作几乎一样，一个是把 预测结果加到 emotion 预测，一个是加到 Cause 预测
        # 在预测 Cause的时候加上 emotion 的 predict logits
        cause_x = torch.cat((cause_x, predict_emotion_logits), dim=-1)

        cause_x, _ = self.cause_lstm(cause_x)
        predict_cause_logits = self.cause_classifier(cause_x)

        return predict_emotion_logits, predict_cause_logits
