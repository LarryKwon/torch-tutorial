import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

x_train = t.FloatTensor([
    [73,80,75],
    [93,88,93],
    [89,91,90],
    [96,98,100],
    [73,66,70]
])
y_train = t.FloatTensor([[152],[185],[180],[196],[142]])

x_test = t.randn((1,3))

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    
    def forward(self,x):
        return self.linear(x)


nb_epochs = 20
model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(1, nb_epochs+1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad() ## gradient 초기화
    cost.backward()
    optimizer.step()
    print(f"grad: {model.linear.weight.grad}")
    print(f"Epoch: {epoch} | y_pred = {prediction.squeeze().detach()} | Loss: {cost.item()}")
    
print(model(x_test))
