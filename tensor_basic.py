import torch
import numpy as np
x = torch.empty(2,2,3)
print(x)

y = torch.rand(2,3)
print(y)
z = torch.ones(2,2,dtype=torch.int16)
print(z)
z = torch.zeros(2,2)
print(z)

ten = torch.rand(5,4)
print(ten[1,1].item())

# Reshape
print(ten.view(4,-1).size())

''' Numpy to tensor and vice versa
if we use CPU ,tensor and numpy use the same memory.Changing in one will effect another
'''
a = torch.rand(4)

b = a.numpy()

a.add_(2)
print(a)
print(b)

c = np.ones(5)
d = torch.from_numpy(c)
print(d)