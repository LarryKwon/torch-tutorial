import torch as t
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
        self.y_data=[[0],[0],[0],[1],[1],[1]]
        
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = t.FloatTensor(self.x_data[index])
        y = t.FloatTensor(self.y_data[index])
        return x,y

dataset = CustomDataset()
dataloader = DataLoader(
    dataset,
    batch_size = 2,
    shuffle=True
)

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        return self.sigmoid(self.linear(x))
    

nb_epochs = 1000
model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr = 1)

for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        loss = F.binary_cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        prediction = hypothesis >= t.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f"Epoch {epoch}/{nb_epochs} Cost: {loss.item()} Accuracy: {accuracy * 100}%")
        
