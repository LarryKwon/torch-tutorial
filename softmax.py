import torch as t
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [
            [1,2,1,1],
            [2,1,3,2],
            [3,1,3,4],
            [4,1,5,5],
            [1,7,5,5],
            [1,2,5,6],
            [1,6,6,6],
            [1,7,7,7]
        ]
        self.y_data=[2,2,2,1,1,1,0,0]
        
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        x = t.FloatTensor(self.x_data[index])
        y = t.LongTensor([self.y_data[index]])
        return x,y

dataset = CustomDataset()
dataloader = DataLoader(
    dataset,
    batch_size = 1,
    shuffle=True
)

class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3)
    
    def forward(self,x):
        return self.linear(x)
    

nb_epochs = 1000
model = SoftmaxClassifier()
optimizer = optim.SGD(model.parameters(), lr = 1)

for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        hypothesis = model(x_train)
        y_train = y_train.squeeze(1)
        # print(y_train)
        loss = F.cross_entropy(hypothesis, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        prediction = t.argmax(hypothesis, dim = 1)
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f"Epoch {epoch}/{nb_epochs} Cost: {loss.item()} Accuracy: {accuracy * 100}%")
        
