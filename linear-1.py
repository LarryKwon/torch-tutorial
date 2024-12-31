import torch as t
import torch.optim as optim

x_train = t.FloatTensor([
    [73,80,75],
    [93,88,93],
    [89,91,90],
    [96,98,100],
    [73,66,70]
])
y_train = t.FloatTensor([[152],[185],[180],[196],[142]])

x_test = t.randn((1,3))

w = t.zeros((3,1),requires_grad=True)
b = t.zeros(1,requires_grad=True)

def forward(x):
    return t.mm(x,w) + b

def loss(y_pred, y_val):
    return t.mean(((y_val - y_pred)**2))

nb_epochs = 20
optimizer = optim.SGD([w,b], lr=1e-5)

for epoch in range(1, nb_epochs+1):
    y_pred = forward(x_train)
    l = loss(y_pred, y_train)
    optimizer.zero_grad() ## gradient 초기화
    l.backward()
    optimizer.step()
    print(f"grad: {w.grad.data}")
    print(f"Epoch: {epoch} | Loss: {l.item()}")

print(forward(x_test))