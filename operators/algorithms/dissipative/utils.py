
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# shape is the tuple shape of each instance
def sample_uniform_spherical_shell(npoints: int, radii: float, shape: tuple):
    ndim = np.prod(shape)
    inner_radius, outer_radius = radii
    pts = []
    for i in range(npoints):
        # uniformly sample radius
        samp_radius = np.random.uniform(inner_radius, outer_radius)
        vec = np.random.randn(ndim) # ref: https://mathworld.wolfram.com/SpherePointPicking.html
        vec /= np.linalg.norm(vec, axis=0)
        pts.append(np.reshape(samp_radius*vec, shape))

    return np.array(pts)

# Partitions of unity - input is real number, output is in interval [0,1]
"""
norm_of_x: real number input
shift: x-coord of 0.5 point in graph of function
scale: larger numbers make a steeper descent at shift x-coord
"""
def sigmoid_partition_unity(norm_of_x, shift, scale):
    return 1/(1 + torch.exp(scale * (norm_of_x - shift)))

# Dissipative functions - input is point x in state space (practically, subset of R^n)
"""
inputs: input point in state space
scale: real number 0 < scale < 1 that scales down input x
"""
def linear_scale_dissipative_target(inputs, scale):
    return scale * inputs

"""
Outputs prediction after post-processing according to:
    rho(|x|) * model(x) + (1 - rho(|x|)) * diss(x)

x: input point as torch tensor
model: torch model
rho: partition of unity, a map from R to [0,1]
diss: baseline dissipative map from R^n to R^n
"""
def part_unity_post_process(x, model, rho, diss):
    return rho(torch.norm(x)) * model(x).reshape(x.shape[0],) + (1 - rho(torch.norm(x))) * diss(x)


# %%
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss
