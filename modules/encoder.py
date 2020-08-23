import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    """
    Section 3.1.1 Encoder
    """

    def __init__(self, name='vgg16', weights_loc=None, out_size=16):
        super().__init__()

        if name=='vgg16':
            model = torchvision.models.vgg16(pretrained=(weights_loc is None))
            cnn_layer = -2
        if weights_loc is not None:
            model.load_state_dict(torch.load(weights_loc))

        modules = list(model.children())[:cnn_layer]
        self.model = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_size, out_size))

    def forward(self, x):
        out = self.model(x)
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out
