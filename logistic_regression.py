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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LogisticRegression(nn.Module):

    def __init__(self, n_input_features, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, output_size)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred



# 0) prepare data
breast_cancer = datasets.load_breast_cancer()
X, y=  breast_cancer.data, breast_cancer.target


n_samples, n_features = X.shape
#print(f'X_SHAPE:{X.shape}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# cast to torch tensor
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model
input_size = n_features
output_size = 1
# f = wx + b, sigmoid function 

model = LogisticRegression(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion =  nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
progress_bar = tqdm(range(num_epochs), desc="Training Progress", total=num_epochs)

for epoch in range(num_epochs):
    y_pedicted = model(X_train)
    loss = criterion(y_pedicted, y_train)
    

    # backward pass
    loss.backward()
    # updates
    optimizer.step()
    # empty grad
    optimizer.zero_grad()

    progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    progress_bar.update(1)

with torch.no_grad():
    y_pedicted = model(X_test)
    y_pedicted_cls = y_pedicted.round()
    # accuracy
    acc = y_pedicted_cls.eq(y_test).sum()/ float(y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')

    