from typing import Tuple

from vit_scratch.models.base_model import BaseModel
import torch


class VitEmbeddingLayer(torch.nn.Module):
    def __init__(
            self,
            sequence_length: int,
            latent_dim: int,
            n_channels: int,
            patch_size: int,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.patch_size = patch_size
        # Create embedding parameters: (N x PPC x D)
        embeddings_shape = (self.sequence_length - 1, (self.patch_size ** 2) * self.n_channels, self.latent_dim)
        self.embeddings = torch.nn.Parameter(torch.rand(embeddings_shape))
        # Create embedding positions parameters: (N x D)
        embeddings_pos_shape = (self.sequence_length - 1, self.latent_dim)
        self.embeddings_pos = torch.nn.Parameter(torch.rand(embeddings_pos_shape))
        # CLS token
        self.cls_embeddings = torch.nn.Parameter(torch.rand(1, *embeddings_pos_shape[1:]))
        self.cls_embeddings_pos = torch.nn.Parameter(torch.rand(1, *embeddings_pos_shape[1:]))

    def forward(self, x: torch.Tensor):
        # Reshape the x tensor in order to perform matmul with the embeddings
        x = x.unsqueeze(2)
        # Perform matrix multiplication to compute the embeddings
        x = x @ self.embeddings
        # Squeeze the result
        x = x.squeeze()
        # Add the positional embeddings
        x = x + self.embeddings_pos
        # Include the CLS token
        cls_token = self.cls_embeddings + self.cls_embeddings_pos
        cls_token = cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        # Return the CLS token concatenated with the embeddings
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(
            self,
            latent_dim: int,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        # Query
        self.q = torch.nn.Linear(self.latent_dim, self.latent_dim)
        # Key
        self.k = torch.nn.Linear(self.latent_dim, self.latent_dim)
        # Value
        self.v = torch.nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, x):
        # Compute Q, K, V
        q, k, v = self.q(x), self.k(x), self.v(x)
        # Matmul qk
        x = q @ k.transpose(dim0=1, dim1=2)
        # Rescale attention
        x = x / torch.sqrt(torch.tensor(self.latent_dim))
        # Compute attention mask
        x = torch.softmax(x, dim=2)
        # Rescale v based on the attention mask
        x = x @ v
        # Return the Scaled Dot-Product Attention
        return x


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(
            self,
            n_heads: int,
            latent_dim: int,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        # List of heads
        self.heads = [AttentionBlock(self.latent_dim) for _ in range(self.n_heads)]
        # Projection layer
        self.projection = torch.nn.Linear(self.latent_dim * self.n_heads, self.latent_dim, bias=False)

    def forward(self, x):
        # Compute attention for each head
        x = [head(x) for head in self.heads]
        # Concatenate attention heads
        x = torch.concat(x, dim=2)
        # Projection
        x = self.projection(x)
        # Return the Multi-Head attention
        return x


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
            self,
            n_heads: int,
            latent_dim: int,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        # Layer norm 1
        self.layer_norm_1 = torch.nn.LayerNorm(self.latent_dim)
        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttentionBlock(self.n_heads, self.latent_dim)
        # Layer Norm 2
        self.layer_norm_2 = torch.nn.LayerNorm(self.latent_dim)
        # MLP Layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * 2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim * 2, latent_dim, bias=True),
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
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        # Create Transformer Encoder Layers
        self.encoder_layers = [TransformerEncoderLayer(n_heads, latent_dim) for _ in range(n_layers)]
        self.encoder = torch.nn.Sequential(*self.encoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class ViTNetwork(torch.nn.Module):

    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            patch_size: int,
            latent_dim: int,
            n_layers: int,
            n_heads: int,
            n_classes: int,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_classes = n_classes
        # Compute sequence length
        self.sequence_length = self._sequence_length_factory(self.input_shape, self.patch_size)
        # Get number of channels
        self.n_channels = self.input_shape[0]
        # Network layers
        self.embeddings_layer: torch.nn.Module = VitEmbeddingLayer(
            sequence_length=self.sequence_length,
            latent_dim=self.latent_dim,
            n_channels=self.n_channels,
            patch_size=self.patch_size
        )
        # Transformer encoder
        self.transformer_encoder: torch.nn.Module = TransformerEncoder(
            n_layers=self.n_layers,
            latent_dim=self.latent_dim,
            n_heads=self.n_heads,
        )
        # MLP Head
        self.mlp_classification_head: torch.nn.Module = torch.nn.Linear(self.latent_dim, self.n_classes)

    @staticmethod
    def _sequence_length_factory(input_shape: Tuple[int, int, int], patch_size: int):
        if input_shape[1] % patch_size != 0 or input_shape[2] % patch_size != 0:
            raise ValueError(f'Input shape {input_shape} not compatible with patch size {input_shape}')
        # Include the "CLS" token
        seq_len = (input_shape[1] * input_shape[2] // patch_size ** 2) + 1
        # Return sequence length
        return seq_len

    def _patchify_and_flatten_image(self, x):
        # Split the image into: N x P^2*C
        size = stride = self.patch_size
        # Patchify the image: B x C x H x W ---> B x C x (H // P) x (W // P) x P x P
        x = x.unfold(2, size, stride).unfold(3, size, stride)
        # Patches tensor shape: B x C x Size x Stride x P x P
        b, c, size, stride, p, _ = x.shape
        # Permute tensor: B x C x Size x Stride x P x P ---> B x Size x Stride x P x P x C
        x = x.permute(0, 2, 3, 4, 5, 1)
        # Unroll tensor: B x Size x Stride x P x P x C ---> B x Size x Stride x P * P * C
        x = x.reshape(b, size, stride, p**2 * c)
        # Flatten the tensor
        x = x.reshape(b, size * stride, p**2 * c)
        # Return the final tensor after patchify() and flatten()
        return x

    def forward(self, x):
        # Patchify the image
        x = self._patchify_and_flatten_image(x)
        # Compute the embeddings
        x = self.embeddings_layer(x)
        # Encoder Layer
        x = self.transformer_encoder(x)
        # Get CLS token
        x = x[:, 0, :]
        # Classification head
        x = self.mlp_classification_head(x)
        # Return the logits
        return x


class ViTCustom(BaseModel):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            patch_size: int,
            latent_dim: int,
            n_layers: int,
            n_heads: int,
            n_classes: int,
            *args, **kwargs
    ):
        vit_network = ViTNetwork(
            input_shape=input_shape,
            patch_size=patch_size,
            latent_dim=latent_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_classes=n_classes,
        )
        super().__init__(vit_network, n_classes, *args, **kwargs)


if __name__ == '__main__':
    input_shape = (1, 28, 28)
    n_classes = 10
    vit_net = ViTNetwork(
        input_shape=input_shape,
        patch_size=7,
        latent_dim=32,
        n_layers=4,
        n_heads=3,
        n_classes=n_classes
    )
    # Crete a dummy input
    batch_size = 10
    dummy_input = torch.rand(batch_size, *input_shape)
    # Feed the input into the network
    x = vit_net(dummy_input)
    # Result shape
    batch_size_pred, n_classes_pred = x.shape
    assert batch_size == batch_size_pred
    assert n_classes == n_classes_pred
    print(x.shape)
