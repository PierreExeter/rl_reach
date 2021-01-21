""" Testing Pytorch and CUDA """

from __future__ import print_function
import torch


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.ones(5, 5)
print(x)

if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    # or just use strings ``.to("cuda")``
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
