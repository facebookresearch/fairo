import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .dmp import DMPIntegrator, DMPParameters


def _linear(in_dim, out_dim, gain=1):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight.data, gain=gain)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class VGGSoftmax(nn.Module):
    def __init__(self, bias=None):
        super().__init__()
        c1, a1 = nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c2, a2 = nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        m1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        c3, a3 = nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c4, a4 = nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        self.vgg = nn.Sequential(c1, a1, c2, a2, m1, c3, a3, c4, a4)
        self.extra_convs = nn.Conv2d(128, 128, 3, stride=2, padding=1)

    def forward(self, x):
        # vgg convs and 2D softmax
        x = self.vgg(x)
        x = self.extra_convs(x)
        B, C, H, W = x.shape
        x = F.softmax(x.view((B, C, H * W)), dim=2).view((B, C, H, W))

        # calculate and return expected keypoints
        h = torch.linspace(-1, 1, H).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 3)
        w = torch.linspace(-1, 1, W).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 2)
        return torch.cat([torch.sum(a, 2) for a in (h, w)], 1)


class PointPredictor(nn.Module):
    def __init__(self, feature_dim, bias=None, hidden_dim=16):
        super().__init__()
        fc1, a1 = nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)
        fc2 = nn.Linear(hidden_dim, 3, bias=False)
        self.top = nn.Sequential(fc1, a1, fc2)
        bias = np.zeros(3).astype(np.float32) if bias is None else np.array(bias).reshape(3)
        self.register_parameter('bias', nn.Parameter(torch.from_numpy(bias).float(), requires_grad=True))
    
    def forward(self, x):
        return self.top(x) + self.bias


class CNNPolicy(nn.Module):
    def __init__(self, features, adim=7, H=1):
        super().__init__()
        self._features = features
        f1, a1 = _linear(256, 256), nn.Tanh()
        f2 = _linear(256, adim * H)
        self._pi = nn.Sequential(f1, a1, f2)
        self._adim, self._H = adim, H
    
    def forward(self, images, _):
        feat = self._features(images)
        if self._H > 1:
            return self._pi(feat).reshape((-1, self._H, self._adim))
        return self._pi(feat).reshape((-1, self._adim))


class RNNPolicy(nn.Module):
    def __init__(self, features, adim=7, H=None):
        super().__init__()
        self._features = features
        self._rnn = nn.LSTM(256, 256, 1, batch_first=True)
        self._pi = nn.Linear(256, adim)
    
    def forward(self, images, _, memory=None, ret_mem=False):
        if len(images.shape) == 5:
            B, T, C, H, W = images.shape
            feat = self._features(images.reshape((B * T, C, H, W))).reshape((B, T, 256))
        else:
            feat = self._features(images)[:,None]
        
        feat, memory = self._rnn(feat, memory)
        feat = feat if len(images.shape) == 5 else feat[:,-1]
        if ret_mem:
            return self._pi(feat), memory
        return self._pi(feat)


class CNNGoalPolicy(nn.Module):
    def __init__(self, features, adim=7, H=1):
        super().__init__()
        self._features = features
        f1, a1 = _linear(512, 256), nn.Tanh()
        f2 = _linear(256, adim * H)
        self._pi = nn.Sequential(f1, a1, f2)
        self._adim, self._H = adim, H
    
    def forward(self, images, goal, _):
        feat = torch.cat((self._features(images), self._features(goal)), 1)
        return self._pi(feat).reshape((-1, self._H, self._adim))


class DMPNet(nn.Module):
    def __init__(self, features, N=270, T=300, tau=1, rbf='gaussian', a_z=15, adim=7, scale=5):
        super().__init__()
        
        output_size = (N+1) * adim
        self.DMPparam = DMPParameters(N, tau, tau / float(T), adim, None, a_z=a_z)
        self.func = DMPIntegrator(rbf=rbf, only_g=False, az=False)
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        self.features = features
        self.fc_last = _linear(256, output_size, gain=scale)
        self._adim = adim
        
   
    def forward(self, images, robot_state):
        h = self.features(images)
        output = self.fc_last(h)
        y0 = robot_state[:, :self._adim].reshape(-1)
        dy0 = torch.ones_like(y0).to(robot_state.device) * 0.05
        y, _, __ = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)
        y = y.view(images.shape[0], self._adim, -1)
        return y.transpose(1, 2)[:,1:]


def restore_pretrain(model):
    # only import torchvision here since it's not in polymetis env 
    from torchvision import models
    if isinstance(model, VGGSoftmax):
        pt = models.vgg16(pretrained=True).features[:10]
        model.vgg.load_state_dict(pt.state_dict())
        return model
    raise NotImplementedError
