import torch
import numpy

# Basic tensors
# a = torch.empty(12,14)
# print(a.shape)


# x= torch.rand(2,2)
# y= torch.rand(2,2)

# z= torch.add(x,y)
# print(z)
# print()
# print(y.add_(x))

# # Reshape using view method
# print()
# k = y.view(-1)
# print(k)

a = torch.ones(5)
b = a.numpy()
print(type(b))
print(type(a))


# Numpy only works on CPU variable
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x= torch.ones(5, device=device)
#     y = torch.ones(5)
#     y=y.to(device)
#     z= x+y
#     z=z.to("cpu")



# Gradient calculation
x =torch.ones(5, requires_grad=True)
print(x)