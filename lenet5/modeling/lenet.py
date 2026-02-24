import math

import torch
from torch import nn
import torch.nn.functional as F

def activation(x):
    return torch.tanh(2/3 * x) / math.tanh(2/3)

class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        m = 2.4/257
        self.w = nn.Parameter((torch.rand(16*16,10)*2-1)*m)
        self.b = nn.Parameter((torch.rand(10)*2-1)*m)
    
    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, 1, 16, 16)
        
        x_flat = x.view(d, 256)
        result = activation(x_flat @ self.w + self.b.view(1, 10))
        assert result.shape == (d, 10)
        
        return result


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        m1 = 2.4/257
        self.w1 = nn.Parameter((torch.rand(16*16,12)*2-1)*m1)
        self.b1 = nn.Parameter((torch.rand(12)*2-1)*m1)

        m2 = 2.4/13
        self.w2 = nn.Parameter((torch.rand(12,10)*2-1)*m2)
        self.b2 = nn.Parameter((torch.rand(10)*2-1)*m2)
    
    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, 1, 16, 16)
        
        x_flat = x.view(d, 256)
        result = activation(x_flat @ self.w1 + self.b1.view(1, 12))
        assert result.shape == (d, 12)

        result = activation(result @ self.w2 + self.b2.view(1, 10))
        assert result.shape == (d, 10)
        
        return result


class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        m1 = 2.4/10
        self.w1 = nn.Parameter((torch.rand(8, 8, 3, 3)*2-1)*m1)
        self.b1 = nn.Parameter((torch.rand(8, 8)*2-1)*m1)

        m2 = 2.4/26
        self.w2 = nn.Parameter((torch.rand(4, 4, 5, 5)*2-1)*m2)
        self.b2 = nn.Parameter((torch.rand(4, 4)*2-1)*m2)

        m3 = 2.4/17
        self.w3 = nn.Parameter((torch.rand(4*4, 10)*2-1)*m3)
        self.b3 = nn.Parameter((torch.rand(10)*2-1)*m3)
    
    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, 1, 16, 16)

        x_padded = F.pad(x, (1, 0, 1, 0), value=-1.)
        assert x_padded.shape == (d, 1, 17, 17)

        conv1 = activation((x_padded.unfold(3,3,2).unfold(2,3,2) * self.w1.view(1, 1, 8, 8, 3, 3)).sum((4,5)) + self.b1.view(1, 1, 8, 8))
        assert conv1.shape == (d, 1, 8, 8)

        conv1_padded = F.pad(conv1, (1, 2, 1, 2), value=-1.)
        assert conv1_padded.shape == (d, 1, 11, 11)

        conv2 = activation((conv1_padded.unfold(3,5,2).unfold(2,5,2) * self.w2.view(1, 1, 4, 4, 5, 5)).sum((4,5)) + self.b2.view(1, 1, 4, 4))
        assert conv2.shape == (d, 1, 4, 4)

        result = activation(conv2.view(d, 4*4) @ self.w3 + self.b3.view(1, 10))
        assert result.shape == (d, 10)
        
        return result


class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        m1 = 2.4/10
        self.w1 = nn.Parameter((torch.rand(2, 1, 3, 3)*2-1)*m1)
        self.b1 = nn.Parameter((torch.rand(2, 8, 8)*2-1)*m1)

        m2 = 2.4/26
        self.w2 = nn.Parameter((torch.rand(2, 4, 4, 5, 5)*2-1)*m2)
        self.b2 = nn.Parameter((torch.rand(4, 4)*2-1)*m2)

        m3 = 2.4/17
        self.w3 = nn.Parameter((torch.rand(4*4, 10)*2-1)*m3)
        self.b3 = nn.Parameter((torch.rand(10)*2-1)*m3)
    
    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, 1, 16, 16)

        x_padded = F.pad(x, (1, 0, 1, 0), value=-1.)
        assert x_padded.shape == (d, 1, 17, 17)

        conv1 = activation(F.conv2d(x_padded, self.w1, padding=0, stride=2) + self.b1.view(1, 2, 8, 8))
        assert conv1.shape == (d, 2, 8, 8)

        conv1_padded = F.pad(conv1, (1, 2, 1, 2), value=-1.)
        assert conv1_padded.shape == (d, 2, 11, 11)

        conv2 = activation((conv1_padded.unfold(3,5,2).unfold(2,5,2) * self.w2.view(1, 2, 4, 4, 5, 5)).sum((1,4,5)).reshape(d, 1, 4, 4) + self.b2.view(1, 1, 4, 4))
        assert conv2.shape == (d, 1, 4, 4)

        result = activation(conv2.view(d, 4*4) @ self.w3 + self.b3.view(1, 10))
        assert result.shape == (d, 10)
        
        return result


class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        m1 = 2.4/10
        self.w1 = nn.Parameter((torch.rand(2, 1, 3, 3)*2-1)*m1)
        self.b1 = nn.Parameter((torch.rand(2, 8, 8)*2-1)*m1)

        m2 = 2.4/26
        self.w2 = nn.Parameter((torch.rand(4, 2, 5, 5)*2-1)*m2)
        self.b2 = nn.Parameter((torch.rand(4, 4, 4)*2-1)*m2)

        m3 = 2.4/17
        self.w3 = nn.Parameter((torch.rand(4*4*4, 10)*2-1)*m3)
        self.b3 = nn.Parameter((torch.rand(10)*2-1)*m3)
    
    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, 1, 16, 16)

        x_padded = F.pad(x, (1, 0, 1, 0), value=-1.)
        assert x_padded.shape == (d, 1, 17, 17)

        conv1 = activation(F.conv2d(x_padded, self.w1, padding=0, stride=2) + self.b1.view(1, 2, 8, 8))
        assert conv1.shape == (d, 2, 8, 8)

        conv1_padded = F.pad(conv1, (1, 2, 1, 2), value=-1.)
        assert conv1_padded.shape == (d, 2, 11, 11)

        conv2 = activation(F.conv2d(conv1_padded, self.w2, padding=0, stride=2) + self.b2.view(1, 4, 4, 4))
        assert conv2.shape == (d, 4, 4, 4)

        result = activation(conv2.view(d, 4*4*4) @ self.w3 + self.b3.view(1, 10))
        assert result.shape == (d, 10)
        
        return result
