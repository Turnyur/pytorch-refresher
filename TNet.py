import torch
import torch.nn as nn
from tqdm import tqdm


class TNet(nn.Module):
    def __init__(self, fan_in, fan_out) -> None:
        super(TNet, self).__init__()
        self.linear = nn.Linear(fan_in, fan_out)

    def forward(self, X:torch.tensor):
        return self.linear(X)
    
    def backprop(self, loss):
        loss.backward()





X_train = torch.tensor([range(1,101), range(1,101)], dtype=torch.float32).T
Y_train = torch.tensor([range(2, 2*101, 2), range(2, 2*101, 2)], dtype=torch.float32).T

X_test = torch.tensor([[5], [5]], dtype=torch.float).T
in_features = X_train.shape[1]
output_size = in_features


model = TNet(in_features, output_size)

# Define loss function and optimizer
learning_rate = 0.000001
n_iters = 15000


progress_bar = tqdm(range(n_iters), desc="Training Progress", total=n_iters)

loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)



prediction = model(X_test)
print(f'Prediction before training: f(X_test) = {prediction.view(-1).tolist()}')

# Training loop
for epoch in range(n_iters):
    # Forward pass
    y_hat = model(X_train)
    
    # Compute loss
    loss = loss_func(y_hat, Y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Optimization step
    optimizer.step()

    progress_bar.set_description(f'Epoch [{epoch+1}/{n_iters}], Loss: {loss.item():.4f}')
    progress_bar.update(1)



prediction = model(X_test)
print("\n\n\n")
print(f'Prediction after training: f(X_test) = {prediction.view(-1).tolist()}')
