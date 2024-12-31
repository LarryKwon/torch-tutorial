import torch
import numpy as np

# x_data = [1.0,2.0,3.0]
x_data = torch.from_numpy(np.array([1.0,2.0,3.0]))
y_data = torch.from_numpy(np.array(object=[2.0, 4.0, 6.0]))
bias = torch.tensor([1.0])
w1 = torch.randn(1, requires_grad=True)
w2 = torch.randn(1, requires_grad=True)
# w = torch.tensor([1.0], requires_grad=True)

def forward(x):
    return (x**2) * w2 + x*w1 + bias

def loss(y_pred, y_val):
    return ((y_pred - y_val) ** 2).mean()

print("Prediction (before training)", 4, forward(4).item())

for epoch in range(10):
    y_pred = forward(x_data)
    l = loss(y_pred, y_data)
    l.backward()
    print(f"grad(w1): {w1.grad.item()}, grad(w2): {w2.grad.item()}")
    w1.data = w1.data - 0.01 * w1.grad.item()
    w2.data = w2.data - 0.01 * w2.grad.item()
    
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    print(f"Epoch: {epoch} | Loss: {l.item()}")

print("Prediction (after training)", 4, forward(4).item())