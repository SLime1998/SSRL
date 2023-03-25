import torch.nn.functional as F
import torch
from torch import nn

def loss_fn(x, y):
     x = F.normalize(x, dim=-1, p=2)
     y = F.normalize(y, dim=-1, p=2)
     # print(x.shape)
     # return -2 * (x * y).sum(dim=-1)
     loss = torch.nn.MSELoss(reduction='none')
     return loss(x, y).sum(dim = -1)

def loss_fn_2(x, y):
    return 2-2 * ((x * y).sum(dim=-1) / (x.norm(dim=-1) * y.norm(dim=-1)))

def loss_fn3(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

x = torch.randn(1, 4)
y = torch.randn(1, 4)

xn = F.normalize(x, dim=-1, p=2)
yn = F.normalize(y, dim=-1, p=2)
# print(loss_fn_2(x, y) - loss_fn_2(xn, yn), "here")

l1 = loss_fn(x, y)
l2 = loss_fn_2(x, y)
l3 = -2 * nn.CosineSimilarity(dim=-1)(x, y)
l4 = loss_fn3(x, y)



print(l1, l2, l3, l4)
t = torch.transpose((x - y), 1, 0)
# print(t.shape, (x- y).shape)
k = F.normalize((x - y), dim=-1, p=2)
print(loss_fn3(x, y) - loss_fn3(xn, yn))
print(F.normalize(x, dim=-1, p=2) - F.normalize(x / 0.5, dim=-1, p=2))