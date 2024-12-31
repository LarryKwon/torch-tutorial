import torch as t
import torch.optim as optim

x_train = t.FloatTensor([[1],[2],[3]])
y_train = t.FloatTensor([[2],[4],[6]])

x_test = t.randn(3)

w = t.zeros(1,requires_grad=True)
b = t.zeros(1,requires_grad=True)

def forward(x):
    return w*x + b

def loss(y_pred, y_val):
    return t.mean(((y_val - y_pred)**2))

nb_epochs = 1000
optimizer = optim.SGD([w,b], lr=0.01)

for epoch in range(1, nb_epochs+1):
    y_pred = forward(x_train)
    l = loss(y_pred, y_train)
    optimizer.zero_grad() ## gradient 초기화
    l.backward()
    optimizer.step()
    print(f"grad: {w.grad.data}")
    print(f"Epoch: {epoch} | Loss: {l.item()}")

print(forward(x_test))