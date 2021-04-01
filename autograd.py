import torch

# Rand generates uniform distribution range[0,1]
x = torch.rand(3,requires_grad=True)
print(x)
# randn generates normal distribution 
y = torch.randn(3)
print(y)

# Stop pytorch tracking history

# x.requires_grad_(False)
# y = x.detach()
# x.detach_()
# with torch.no_grad():
#     print(x+2)

weights = torch.ones(4,requires_grad=True)

for epoch in range(2):
    model_output = (weights*3).sum()

    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()