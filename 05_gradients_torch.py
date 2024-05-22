import torch
import torch.nn as nn
from tqdm import tqdm


# f = w * x

# f = 2 * x
X = torch.tensor([range(1,101), range(1,101)], dtype=torch.float32).T
Y = torch.tensor([range(2, 2*101, 2), range(2, 2*101, 2)], dtype=torch.float32).T

X_test = torch.tensor([[5], [5]], dtype=torch.float).T
in_features = X.shape[1]
output_size = in_features

#print(in_features)

model = nn.Linear(in_features, output_size)

#print(f'X:{X_test.shape}, Y:{Y.shape}')
#exit()
#w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Model prediction

# def forward(x:torch.Tensor):
#     f = w * x
#     return f

# loss = MSE
# def loss(y, y_hat):
#     l = (y_hat - y)**2
#     return l.mean()



# Gradient
# MSE = 1/N * (w*x - y)*2
# dJ/dw =  1/N 2x (w*x - y)


prediction = model(X_test)
print(f'Prediction before training: f(X_test) = {prediction.view(-1).tolist()}')

#print(f'TYPE W: {type(w)}')

# Training


learning_rate = 0.0001
n_iters = 150


progress_bar = tqdm(range(n_iters), desc="Training Progress", total=n_iters)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l=loss(Y, y_pred)


    optimizer.zero_grad()
    # gradients
    l.backward() # dl/dw

    # update weights
    # with torch.no_grad():
    #     w.sub_(learning_rate * w.grad)
    optimizer.step()

    

    #if epoch % 1000 == 0:
    #    [w, b] = model.parameters()
    #    print(f'Epoch [{epoch+1}/{n_iters}], Loss: {l.item():.4f}')
    progress_bar.set_description(f'Epoch [{epoch+1}/{n_iters}], Loss: {l.item():.4f}')
    progress_bar.update(1)


prediction = model(X_test)
print("\n\n\n")
print(f'Prediction after training: f(X_test) = {prediction.view(-1).tolist()}')
