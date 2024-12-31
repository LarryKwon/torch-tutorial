import torch as t
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73,80,75],
                        [93,88,93],
                        [89,91,90],
                        [96,98,100],
                        [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]
        self.x_test = t.randn((1,3))
        
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
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad() ## gradient 초기화
        cost.backward()
        optimizer.step()
        print(f"grad: {model.linear.weight.grad}")
        print(f"Epoch: {epoch}| Batch {batch_idx+1} / {len(dataloader)} | y_pred = {prediction.squeeze().detach()} | Loss: {cost.item()}")
        
print(model(dataset.x_test))
