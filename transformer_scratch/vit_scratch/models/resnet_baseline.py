import torch

from transformer_scratch.vit_scratch.config import ResnetBaselineConfig


class ResnetBaselineNetwork(torch.nn.Module):
    def __init__(self, resnet_config: ResnetBaselineConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize a ResNet backbone
        self.resnet = torch.hub.load(
            'pytorch/vision',
            f'resnet{resnet_config.resnet_size}',
            pretrained=resnet_config.pretrained
        )
        # Modify the last layer for the number of classes in your classification task
        self.resnet.conv1 = torch.nn.Conv2d(
            resnet_config.num_channels,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size[0],
            stride=self.resnet.conv1.stride[0],
            padding=self.resnet.conv1.stride[0],
            bias=False
        )
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, resnet_config.n_classes)

    def forward(self, x):
        return self.resnet(x)
