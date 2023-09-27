from vit_scratch.models.base_model import BaseModel
import torch


class ResnetBaselineNetwork(torch.nn.Module):
    def __init__(
            self,
            num_classes: int,
            num_channels: int,
            resnet_size: int,
            pretrained: bool,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # Initialize a ResNet backbone
        self.resnet = torch.hub.load(
            'pytorch/vision', f'resnet{resnet_size}', pretrained=pretrained)
        # Modify the last layer for the number of classes in your classification task
        self.resnet.conv1 = torch.nn.Conv2d(
            num_channels,
            self.resnet.conv1.out_channels,
            kernel_size=self.resnet.conv1.kernel_size[0],
            stride=self.resnet.conv1.stride[0],
            padding=self.resnet.conv1.stride[0],
            bias=False
        )
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


class ResnetBaseline(BaseModel):
    def __init__(
            self,
            num_classes: int,
            num_channels: int,
            pretrained: bool = False,
            resnet_size: int = 18,
            *args, **kwargs
    ):
        resnet_network = ResnetBaselineNetwork(
            num_classes, num_channels, resnet_size, pretrained)
        super().__init__(resnet_network, num_classes)

