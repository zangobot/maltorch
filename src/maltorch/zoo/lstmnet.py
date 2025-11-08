"""
Learning the PE Header, Malware Detection with Minimal Domain Knowledge
E. Raff, J. Sylvester, C. Nicholas
https://dl.acm.org/doi/10.1145/3128572.3140442
"""
from torch import nn
from maltorch.zoo.model import EmbeddingModel

class LSTMNet(EmbeddingModel):
    def __init__(self,
                embedding_size: int = 16,
                max_len: int = 512,
                min_len: int = 512,
                threshold: float = 0.5,
                padding_idx: int = 256,
                hidden_size: int = 512,
                num_layers: int = 3,
                dropout: float = 0.20
               ):
        super(LSTMNet, self).__init__(
           name="LSTMNet", gdrive_id=None, max_len=max_len, min_len=min_len
        )
        self.seq_len = min_len
        self.threshold = threshold

        # Embedding
        self.embedding_1 = nn.Embedding(
           num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2*hidden_size, 1)

    def embedding_layer(self):
        return self.embedding_1

    def embed(self, x):
        emb_x = self.embedding_1(x)
        emb_x = emb_x.transpose(1, 2)
        return emb_x

    def embedding_matrix(self):
        return self.embedding_1.weight

    def _forward_embed_x(self, x):
        """
        x: (B, T) token indices
        lengths: (B,) true lengths before padding
        """
        #emb = self.emb(x)  # (B, T, E)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        pooled, _ = lstm_out.max(dim=1)  # (B, 2H)
        logits = self.fc(self.dropout(pooled))  # (B, num_classes)
        return logits
