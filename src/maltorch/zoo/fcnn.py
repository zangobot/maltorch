"""
Learning the PE Header, Malware Detection with Minimal Domain Knowledge
E. Raff, J. Sylvester, C. Nicholas
https://dl.acm.org/doi/10.1145/3128572.3140442
"""
from torch import nn
from maltorch.zoo.model import EmbeddingModel


class FCNN(EmbeddingModel):

    def __init__(self,
                 embedding_size: int = 16,
                 max_len: int = 512,
                 min_len: int = 512,
                 threshold: float =0.5,
                 padding_idx: int = 256,
                 hidden_dims: tuple = (512, 512, 512, 512),
                 p_embed_dropout: float = 0.20,
                 p_hidden_dropout: float = 0.50):
        super(FCNN, self).__init__(
           name="FCNN", gdrive_id=None, max_len=max_len, min_len=min_len
        )
        self.embedding_1 = nn.Embedding(
           num_embeddings=257, embedding_dim=embedding_size, padding_idx=padding_idx
        )
        self.embedding_1_dropout = nn.Dropout(p_embed_dropout)

        in_dim = max_len * embedding_size
        self.dense_1 = nn.Linear(in_features=in_dim, out_features=hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dense_2 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dense_3 = nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dense_4 = nn.Linear(in_features=hidden_dims[2], out_features=hidden_dims[3])
        self.bn4 = nn.BatchNorm1d(hidden_dims[3])

        self.out = nn.Linear(in_features=hidden_dims[3], out_features=1)
        self.dropout = nn.Dropout(p_hidden_dropout)
        self.act = nn.ELU()

        self.embedding_size = (embedding_size,)
        self.max_len = max_len
        self.threshold = threshold
        self.invalid_value = padding_idx
        self.hidden_dims = hidden_dims
        self.p_embed_dropout = p_embed_dropout
        self.p_hidden_dropout = p_hidden_dropout

    def embedding_layer(self):
        return self.embedding_1

    def embed(self, x):
        emb_x = self.embedding_1(x)
        emb_x = emb_x.transpose(1, 2)
        return emb_x

    def _forward_embed_x(self, x):
        x = self.embedding_1_dropout(x)
        z = x.reshape(x.size(0), -1)

        # FC Blocks
        h1 = self.dropout(self.act(self.bn1(self.dense_1(z))))
        h2 = self.dropout(self.act(self.bn1(self.dense_2(h1))))
        h3 = self.dropout(self.act(self.bn1(self.dense_3(h2))))
        h4 = self.dropout(self.act(self.bn1(self.dense_4(h3))))
        out = self.out(h4)
        return out

    def embedding_matrix(self):
        return self.embedding_1.weight





