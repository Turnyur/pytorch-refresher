import torch


arr_list = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]

x = torch.tensor(arr_list, dtype=torch.double, requires_grad=True)
#print(x)

for epoch in range(20):
    y = (x**2)+2

    z = y*0.3

    #print(z)
    dz = z.sum()
    dz.backward()
    print(x.grad)
    x.grad.zero_()
    

