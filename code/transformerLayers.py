import torch
import torch.nn as nn
from networks import Normalization
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair



DEVICE = 'cpu'
INPUT_SIZE = 28
DEBUG=False

# TODO set DEBUG to False

class AffineTransformer(torch.nn.Module):
   # __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, D_features, learnedWeights, learnedBias=None):
        super(AffineTransformer, self).__init__()
        self.in_features = D_features
        self.out_features = D_features
        self.weight = learnedWeights
        self.weight.requires_grad_(False)
        if learnedBias is not None:
            self.bias = learnedBias
        else:
            self.bias=None
        self.bias.requires_grad_(False)

    def forward(self, input):
        A = input[1]
        a0 = input[0]
        A_n=torch.matmul(A,self.weight.T)
        a0_n=torch.matmul(self.weight,a0)+self.bias
        return (a0_n,A_n)

class NormalizationTransformer(torch.nn.Module):
    def __init__(self):
        super(NormalizationTransformer, self).__init__()
        self.mean = torch.FloatTensor([0.1307]).to(DEVICE)
        self.sigma = torch.FloatTensor([0.3081]).to(DEVICE)

    def forward(self, input):
        A=input[1]
        a0=input[0]
        A_n=torch.div(A,self.sigma)
        a0_n=torch.div(a0-self.mean,self.sigma)
        return (a0_n,A_n)

class FlattenTransformer(torch.nn.Module):
    def __init__(self):
        super(FlattenTransformer, self).__init__()

    def forward(self, input):
        A = input[1]
        a0 = input[0]
        A_n=A.flatten(start_dim=1,end_dim=-1)
        a0_n=a0.flatten()
        return (a0_n, A_n)

class ReLuTransformer(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, D_features,x_shape):
        super(ReLuTransformer, self).__init__()
        self.in_features = D_features
        self.out_features = D_features+x_shape
        self.slopes = Parameter(torch.Tensor(x_shape))
        self.slopes.requires_grad_(True)
        self.x_shape=x_shape
        self.initialized=False
        #self.hyper_diag=torch.diagflat(torch.ones(np.prod(x_shape))).view((-1,) + x_shape)

    def reset_parameters(self,A_in):
        A = A_in[1].detach()
        a0 = A_in[0].detach()
        temp_slope = 0.5 * (1.0 + torch.div(a0, torch.abs(A).sum(axis=0)))
        temp_slope = torch.clamp(temp_slope,0,1)
        self.slopes.data=temp_slope

    def forward(self, A_in):
        if not self.initialized:
            ReLuTransformer.reset_parameters(self, A_in)
            self.initialized = True
        A = A_in[1]     # eps x x_shape
        a0 = A_in[0]    # x_shape

        l_x = -torch.abs(A).sum(axis=0) + a0
        u_x =  torch.abs(A).sum(axis=0) + a0

        crossover_idx = torch.mul((l_x < 0).float(), (u_x > 0).float())
        offset_factor = torch.max(u_x*(1 - self.slopes), -l_x*self.slopes) / 2.

        a0_n = torch.mul((1 - crossover_idx), torch.nn.functional.relu(a0)) \
                 + torch.mul(crossover_idx, torch.mul(self.slopes, a0) + offset_factor)

        A_new_e, A_0 = self.assemble_hyper_diagonal(self.x_shape, offset_factor, crossover_idx)

        A_n = torch.mul((l_x > 0).float().unsqueeze(0), torch.cat((A, A_0),dim=0))\
                + torch.mul(crossover_idx.unsqueeze(0), torch.cat((torch.mul(self.slopes.unsqueeze(0), A), A_new_e), dim=0))

        if DEBUG:
            l_x_n = -torch.abs(A_n).sum(axis=0) + a0_n
            u_x_n = torch.abs(A_n).sum(axis=0) + a0_n

            if not (((u_x_n-u_x)/abs((u_x_n-u_x)).median())>-1e-4).all() and (u_x_n-u_x).min()<-1e-5:
                print("ReLU upper bound soundness check failed")
            if not (((l_x_n-l_x)/abs((l_x_n-l_x)).median())>-1e-4).all() and (l_x_n-l_x).min()<-1e-5:
                print("ReLU lower bound soundness check failed")
        return (a0_n,A_n)

    def assemble_hyper_diagonal(self,x_shape,offset_factor,crossover_idx):
        # Build Tensor for new epsilons corresponding to a hyper diagonal in 4D (eps x channel x x_shape) or
        # 2D (eps x x_shape) with offset_factor on the entries indicated by crossover_idx

        '''A_new = torch.zeros((int(crossover_idx.sum()),) + x_shape, dtype=torch.float32)  # Channels x space dim x eps space dim
        A_0=torch.clone(A_new)
        k_i = 0
        if len(x_shape) > 1:
            for i in range(x_shape[0]):
                for j in range(x_shape[1]):
                    for l in np.array(range(x_shape[2]))[crossover_idx[i,j,:]==1]:
                        A_new[k_i, i, j, l] = offset_factor[i, j, l]
                        k_i += 1

        else:
            for i in np.array(range(x_shape[0]))[crossover_idx==1]:
                A_new[k_i, i] = offset_factor[i]
                k_i += 1
        '''
        # A_new=torch.diagflat(offset_factor)[crossover_idx.flatten() == 1, :].view((int(crossover_idx.sum()),) + x_shape)

        k=int(crossover_idx.sum())
        A_new=torch.sparse.FloatTensor(torch.cat([torch.arange(k).view(-1, 1), crossover_idx.nonzero()], dim=1).T,
                                 offset_factor[crossover_idx == 1].flatten(), torch.Size((k,) + x_shape)).to_dense()
        '''if len(x_shape) > 1:
            A_new=torch.mul(self.hyper_diag[crossover_idx.flatten() == 1, :, :, :],
                      offset_factor[crossover_idx == 1].flatten().view((k, 1, 1, 1)), )
        else:
            A_new = torch.mul(self.hyper_diag[crossover_idx.flatten() == 1, :],
                              offset_factor[crossover_idx == 1].flatten().view((k, 1)), )
        '''
        A_0=torch.zeros((int(crossover_idx.sum()),) + x_shape, dtype=torch.float32)

        return A_new, A_0


class Conv2DTransformer(torch.nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, weights, bias, in_channels, out_channels, kernel_size, out_shape, stride=1,
                 padding=0, dilation=1, groups=1,
                 padding_mode='zeros'):
        super(Conv2DTransformer, self).__init__()

        self.a0_conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                               stride=stride,padding=padding,padding_mode=padding_mode,dilation=dilation,
                               groups=groups,bias=True)
        self.a0_conv.weight=weights
        self.a0_conv.weight.requires_grad_(False)
        self.a0_conv.bias = bias
        self.a0_conv.bias.requires_grad_(False)

        self.A_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation,
                                 groups=groups, bias=False)
        self.A_conv.weight = weights
        self.A_conv.weight.requires_grad_(False)

    def forward(self, A_in):
        a0 = A_in[0]
        A = A_in[1]
        a0_n=self.a0_conv(a0.unsqueeze(0))[0,:,:,:]
        A_n= self.A_conv(A)
        return (a0_n, A_n)

class PairwiseDiffTransformer(torch.nn.Module):

    def __init__(self, D_features, true_label):
        super(PairwiseDiffTransformer, self).__init__()
        self.in_features = D_features
        self.true_label = true_label
        self.weight = PairwiseDiffTransformer.getPairwiseDiff(true_label,D_features[0])
        self.weight.requires_grad_(False)

    def forward(self, input):
        A = input[1]
        a0 = input[0]
        A_n = torch.matmul(A,self.weight.T)
        a0_n = self.weight.matmul(a0)
        l_x = -torch.abs(A_n).sum(axis=0) + a0_n
        return l_x

    def getPairwiseDiff(true_label,n_labels):
        A=torch.zeros((n_labels-1,n_labels))
        A[:,true_label]+=1
        A[:true_label, :true_label] -= torch.eye(true_label)
        A[true_label:, true_label + 1:] = -torch.eye(n_labels - true_label - 1)
        return A

def conv2doutput(layer,input_size):
    out_shape=[0,0]
    for j in range(2):
        out_shape[j]=int((input_size[j]+2*layer.padding[j]-layer.dilation[j]*layer.kernel_size[j])/layer.stride[j]+1)
    return tuple(out_shape)

def translate_nn(net,true_label,input_size):
    # Build a network of transformer layers corresponding to the original network
    layers=[]
    x_shape=input_size
    k=np.prod(x_shape)
    for layer in net.layers:
        if isinstance(layer, torch.nn.Linear):
            layers += [AffineTransformer(k,layer._parameters['weight'],layer._parameters['bias'])]
            x_shape=(layer.out_features,)
        elif isinstance(layer, torch.nn.Conv2d):
            x_shape = (layer.out_channels,) + conv2doutput(layer, x_shape[-2:])

            layers += [Conv2DTransformer(layer._parameters['weight'], layer._parameters['bias'],
                                         layer.in_channels,layer.out_channels,layer.kernel_size,out_shape=x_shape,
                                         stride=layer.stride,padding=layer.padding,dilation=layer.dilation,
                                         groups=layer.groups,padding_mode=layer.padding_mode)]

        elif isinstance(layer, torch.nn.ReLU):
            layers += [ReLuTransformer(k,x_shape)]
            k=k+np.prod(x_shape)
        elif isinstance(layer, torch.nn.Flatten):
            x_shape=np.prod(x_shape)
            layers += [FlattenTransformer()]
        elif isinstance(layer, Normalization):
            layers += [NormalizationTransformer()]
        else:
            print("Error - Layer not recognized")
    layers += [PairwiseDiffTransformer(x_shape,true_label)]
    return nn.Sequential(*layers)