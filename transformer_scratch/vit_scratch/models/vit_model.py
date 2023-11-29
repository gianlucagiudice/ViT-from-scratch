from typing import Tuple

import torch

from transformer_scratch import EmbeddingLayer, TransformerEncoder
from transformer_scratch.vit_scratch.config import ViTConfig


class ViTEmbeddingLayer(EmbeddingLayer):
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
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        self.embeddings = torch.nn.init.xavier_uniform_(self.embeddings)
        self.embeddings_pos = torch.nn.init.xavier_uniform_(self.embeddings_pos)
        self.cls_embeddings = torch.nn.init.xavier_uniform_(self.cls_embeddings)
        self.cls_embeddings_pos = torch.nn.init.xavier_uniform_(self.cls_embeddings_pos)

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


class ViTNetwork(torch.nn.Module):

    def __init__(self, vit_config: ViTConfig):
        super().__init__()
        self.input_shape = vit_config.input_shape
        self.patch_size = vit_config.patch_size
        self.latent_dim = vit_config.latent_dim
        self.n_layers = vit_config.n_layers
        self.n_heads = vit_config.n_heads
        self.n_classes = vit_config.n_classes
        self.dropout = vit_config.dropout
        # Compute sequence length
        self.sequence_length = self._sequence_length_factory(self.input_shape, self.patch_size)
        # Get number of channels
        self.n_channels = self.input_shape[0]
        # Network layers
        self.embedding: torch.nn.Module = ViTEmbeddingLayer(
            sequence_length=self.sequence_length,
            latent_dim=self.latent_dim,
            n_channels=self.n_channels,
            patch_size=self.patch_size
        )
        # Transformer encoder
        self.encoder: torch.nn.Module = TransformerEncoder(
            n_layers=self.n_layers,
            latent_dim=self.latent_dim,
            n_heads=self.n_heads,
            dropout=self.dropout,
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
        x = x.reshape(b, size, stride, p ** 2 * c)
        # Flatten the tensor
        x = x.reshape(b, size * stride, p ** 2 * c)
        # Return the final tensor after patchify() and flatten()
        return x

    def forward(self, x):
        # Patchify the image
        x = self._patchify_and_flatten_image(x)
        # Compute the embeddings
        x = self.embedding(x)
        # Encoder Layer
        x = self.encoder(x)
        # Get CLS token
        x = x[:, 0, :]
        # Classification head
        x = self.mlp_classification_head(x)
        # Return the logits
        return x


if __name__ == '__main__':
    n_classes = 10
    batch_size = 10

    # Create a ViT config
    vit_config = ViTConfig(
        input_shape=(1, 28, 28),
        patch_size=7,
        latent_dim=32,
        n_layers=4,
        n_heads=4,
        n_classes=n_classes,
        dropout=0.1,
    )
    # Create a ViT network
    vit_net = ViTNetwork(vit_config)

    # Crete a dummy input
    dummy_input = torch.rand(batch_size, *vit_config.input_shape)
    # Feed the input into the network
    x = vit_net(dummy_input)
    # Result shape
    batch_size_pred, n_classes_pred = x.shape
    assert batch_size == batch_size_pred
    assert n_classes == n_classes_pred
    print(x.shape)
