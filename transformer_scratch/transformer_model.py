import torch


class EmbeddingLayer(torch.nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            sequence_length: int,
            latent_dim: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        # Create embedding parameters: (N x D)
        self.embeddings = torch.nn.Parameter(torch.rand((self.num_embeddings, self.latent_dim)))
        self.embeddings_pos = torch.zeros((self.sequence_length, self.latent_dim))
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        # Embeddings
        self.embeddings = torch.nn.init.xavier_uniform_(self.embeddings)

        # Positional Embeddings
        # P E(pos,2i)   = sin(pos/10000^(2i/dmodel))
        # P E(pos,2i+1) = cos(pos/10000^(2i/dmodel))
        pos_ids = torch.arange(self.sequence_length).unsqueeze(1).repeat(1, self.latent_dim)
        emb_ids = torch.arange(self.latent_dim).unsqueeze(0).repeat(self.sequence_length, 1)

        emb_tmp = pos_ids / (10_000 ** (2 * emb_ids / self.latent_dim))

        even_len = self.latent_dim // 2 + self.latent_dim % 2
        odd_len = self.latent_dim // 2

        even_pos = torch.sin(emb_tmp)[:, :even_len]
        odd_pos = torch.cos(emb_tmp)[:, :odd_len]

        self.embeddings_pos = torch.empty((self.sequence_length, self.latent_dim))
        self.embeddings_pos[:, torch.arange(even_len) * 2] = even_pos
        self.embeddings_pos[:, (torch.arange(odd_len) * 2) + 1] = odd_pos

    def forward(self, x):
        # Get the embeddings indexed at the token ID
        x = self.embeddings[x]
        x = x + self.embeddings_pos
        return x


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(
            self,
            n_heads: int,
            latent_dim: int,
            dropout: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        if self.latent_dim % self.n_heads != 0:
            raise ValueError(f'Latent dim {self.latent_dim} not divisible by n_heads {self.n_heads}')
        self.head_dim = self.latent_dim // self.n_heads
        self.dropout = dropout
        # Multi-head QKV
        self.qkv = torch.nn.Linear(self.latent_dim, self.head_dim * self.n_heads * 3, bias=False)
        # Attention dropout
        self.attn_dropout = torch.nn.Dropout(self.dropout)
        # Projection layer
        self.projection = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)

    def forward(self, x):
        # Compute QKV for each head
        qkv = self.qkv(x)
        # Reshape multi head Q, K, V
        qkv = qkv.view(x.shape[0], x.shape[1], 3, self.n_heads, self.head_dim)
        # Permute the tensor: B x S x 3 x H x D ---> 3 x H x B x S x D
        qkv = qkv.permute(2, 3, 0, 1, 4)
        # Get Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Matmul QK
        attn = q @ k.transpose(dim0=-1, dim1=-2)
        # Rescale attention
        attn = attn / torch.sqrt(torch.tensor(self.head_dim))
        # Compute attention mask
        attn = torch.softmax(attn, dim=-1)
        # Rescale v based on the attention mask
        attn = attn @ v
        # Reshape the attention tensor: H x B x S x D ---> B x S x H x D
        attn = attn.permute(1, 2, 0, 3)
        # Concatenate the multiple heads
        attn = attn.reshape(x.shape[0], x.shape[1], self.n_heads * self.head_dim)
        # Project attention
        attn = self.projection(attn)
        # Dropout layer
        attn = self.attn_dropout(attn)
        # Return the encoded layer
        return attn


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            n_heads: int,
            latent_dim: int,
            dropout: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.dropout = dropout
        # Layer norm 1
        self.layer_norm_1 = torch.nn.LayerNorm(self.latent_dim)
        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttentionBlock(self.n_heads, self.latent_dim, self.dropout)
        # Layer Norm 2
        self.layer_norm_2 = torch.nn.LayerNorm(self.latent_dim)
        # MLP Layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, latent_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        # Save x for residual connection 1
        x_res = x
        # Normalization 1
        x = self.layer_norm_1(x)
        # Multi-Head Attention Block
        x = self.multi_head_attention(x)
        # Apply residual connection 1
        x = x + x_res
        # Save x for residual connection 2
        x_res = x
        # Normalization 2
        x = self.layer_norm_2(x)
        # MLP
        x = self.mlp(x)
        # Apply residual connection 2
        x = x + x_res
        # Return the encoded layer
        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            latent_dim: int,
            dropout: float,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        # Create Transformer Encoder Layers
        self.encoder = torch.nn.Sequential(*[
            TransformerEncoderLayer(n_heads, latent_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.encoder(x)
        return x


class TransformerNetwork:
    pass


if __name__ == '__main__':
    pass
