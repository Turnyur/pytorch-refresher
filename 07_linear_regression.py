# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#    - forward pass: compute prediction and loss
#    - backward pass: gradients
#    - update weights


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
from TNet import TNet as Regression

# 0) prepare data
X_numpy, y_numpy =  datasets.make_regression(n_samples=10000, n_features=1, noise=20, random_state=1)

#print(f'X_SHAPE:{X_numpy.shape}  y_SHAPE:{y_numpy.shape}')

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y= y.view(y.shape[0], 1)

#print(f'\ny_SHAPE:{y_numpy.shape}')
n_samples, n_features = X.shape
# 1) model
input_size = n_features
output_size = 1
#model = nn.Linear(input_size, output_size)
model = Regression(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Trainign Loop

num_epochs = 100
progress_bar = tqdm(range(num_epochs), desc="Training Progress", total=num_epochs)

for epoch in range(num_epochs):
    # forward pass
    y_pred = model(X)
    loss =criterion(y_pred, y)
    # backward pass
    loss.backward()

    # update
    optimizer.step()

    optimizer.zero_grad()

    progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    progress_bar.update(1)

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, color='orange', marker='o', linestyle='', label='Actual')
plt.plot(X_numpy, predicted, color='red', label='Predicted')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.legend()

plt.show()