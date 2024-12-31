import torch as t
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F


x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]

x_train = t.FloatTensor(x_data)
y_train = t.FloatTensor(y_data)

W = t.zeros((2,1),requires_grad=True)
b = t.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    
    hypothesis = t.sigmoid(x_train.matmul(W)+b)
    prediction = hypothesis >= t.FloatTensor([0.5])
    loss = F.binary_cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{nb_epochs} Cost: {loss.item()}")
        
