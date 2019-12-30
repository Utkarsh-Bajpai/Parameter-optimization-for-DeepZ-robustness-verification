import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Normalization(nn.Module):

    def __init__(self, device):
        super(Normalization, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    def forward(self, x):
        return (x - self.mean) / self.sigma


class FullyConnected(nn.Module):

    def __init__(self, device, input_size, fc_layers):
        super(FullyConnected, self).__init__()

        layers = [Normalization(device), nn.Flatten()]
        prev_fc_size = input_size * input_size
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class UnitTest(nn.Module):

    def __init__(self,device):
        super(UnitTest, self).__init__()
        #prev_fc_size = 2
        #fc_layers = 2
        linearLayer1=nn.Linear(4, 2)
        linearLayer1.weight = Parameter(torch.tensor([[-1, -1,1,0], [-6, 0,-1,2]], dtype=torch.float32))
        linearLayer1.bias = Parameter(torch.tensor([0, 0], dtype=torch.float32))
        ReLULayer1=nn.ReLU()
        linearLayer2 = nn.Linear(2, 2)
        linearLayer2.weight = Parameter(torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32))
        linearLayer2.bias = Parameter(torch.tensor([0.2, -0.1], dtype=torch.float32))
        self.layers = nn.Sequential(*[nn.Flatten(), linearLayer1,ReLULayer1,linearLayer2])

    def forward(self, x):
        return self.layers(x)

class Conv(nn.Module):

    def __init__(self, device, input_size, conv_layers, fc_layers, n_class=10):
        super(Conv, self).__init__()

        self.input_size = input_size
        self.n_class = n_class

        layers = [Normalization(device)]
        prev_channels = 1
        img_dim = input_size

        for n_channels, kernel_size, stride, padding in conv_layers:
            layers += [
                nn.Conv2d(prev_channels, n_channels, kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
            ]
            prev_channels = n_channels
            img_dim = img_dim // stride
        layers += [nn.Flatten()]

        prev_fc_size = prev_channels * img_dim * img_dim
        for i, fc_size in enumerate(fc_layers):
            layers += [nn.Linear(prev_fc_size, fc_size)]
            if i + 1 < len(fc_layers):
                layers += [nn.ReLU()]
            prev_fc_size = fc_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
