import torch
import numpy as np


# f = w * x

# f = 2 * x
X = np.array(range(1,101), dtype=np.float32)
Y = np.array(range(2, 2*101, 2), dtype=np.float32)


#print(f'X:{X.shape}, Y:{Y.shape}')
w = 0.0

# Model prediction

def forward(x):
    f = w * x
    return f

# loss = MSE
def loss(y, y_hat):
    l = (y_hat - y)**2
    return l.mean()



# Gradient
# MSE = 1/N * (w*x - y)*2
# dJ/dw =  1/N 2x (w*x - y)
def gradient(x, y_hat, y):
    dj_dw =  np.dot(2*x, (y_hat-y)).mean()
    return dj_dw


print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Training

learning_rate = 0.0000001
n_iters = 150

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l=loss(Y, y_pred)

    # gradients
    dl_dw = gradient(X,y_pred, Y)

    # update weights
    w = w - learning_rate * dl_dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')