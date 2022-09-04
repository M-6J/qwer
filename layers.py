import torch
import torch.nn as nn
#tdbn - https://paperswithcode.com/paper/going-deeper-with-directly-trained-larger#code
#https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
#https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/spiking_resnet.py
#https://github.com/fangwei123456/spikingjelly/blob/940737e6f25687b1b93de41291603fa875e471f5/spikingjelly/activation_based/layer.py
#https://github.com/fangwei123456/spikingjelly/blob/940737e6f25687b1b93de41291603fa875e471f5/spikingjelly/activation_based/functional.py#L624
class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        #x_seq.shape[0] = 16, x_seq.shape[1] = 2
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        mem = 0
        spike_pot = []
        T = x.shape[1]
        for t in range(T):
            #TET Eq.1
            mem = mem * self.tau + x[:, t, ...]
            #TET Eq.2
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            #TET Eq.3
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        return torch.stack(spike_pot, dim=1)


class ZIF(torch.autograd.Function): #TET Eq.4
    @staticmethod
    def forward(ctx, input, gama): 
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        #TET Eq.5
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None



def add_dimention(x, T):
    x.unsqueeze_(1) # torch.Size([16, 1, 3, 32, 32])
    x = x.repeat(1, T, 1, 1, 1) # torch.Size([16, 2, 3, 32, 32])
    return x


class Layer(nn.Module):
    def __init__(self,in_plane,out_plane,kernel_size,stride,padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
            nn.BatchNorm2d(out_plane)
        )
        self.act = LIFSpike()

    def forward(self,x):
        x = self.fwd(x)
        x = self.act(x)
        return x

class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y

class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_