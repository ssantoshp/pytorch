import torch
import torch.nn as nn # to build our neural net + loss funcs
import torch.optim as optim # optimization alog like adams, gradient descent
import torch.nn.functional as F # stuff like relu, func that do not have params
from torch.utils.data import DataLoader # helps us create mini batches on data
import torchvision.datasets as datasets # pytorch built in datasets
import torchvision.transforms as transforms # transformation we can do on our dataset

# create Fully Connected Neural Network
class NN(nn.Module): # inherits from nn.Module
    def __init__(self, input_size, num_classes): # (28x28) in MNIST
        super(NN, self).__init__() # run initalization of that method
        self.fc1 = nn.Linear(input_size, 50) # 50 hidden nodes
        self.fc2 = nn.Linear(50, num_classes) # easy

    def forward(self, x):
       x =  F.relu(self.fc1(x)) # store in x result after applying relu on fc1
       x = self.fc2(x) # pass x (relu applied) to second layer and store its output in x
       return x # return pred


class CNN(nn.Module):
  def __init__(self, in_channels = 1, num_classes = 10): # =1 for MNIST
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels = 8, kernel_size=(3, 3), stride = (1, 1), padding=(1, 1))
    self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2, 2))
    self.conv2 = nn.Conv2d(in_channels=8, out_channels = 16, kernel_size=(3, 3), stride = (1, 1), padding=(1, 1))
    self.fc1 = nn.Linear(16*7*7, num_classes)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(x)
      x = F.relu(self.conv2(x)) 
      x = self.pool(x)
      x = x.reshape(x.shape[0], -1) #  flatten
      x = self.fc1(x)
      return x


model = CNN()

'''
model = NN(784, 10) # 28*28 = 784, 10 digits/classes
x = torch.randn(64, 784) # 64 examples
print(model(x).shape) # we want output to be 64x10, prob of each classes for each example
# out: torch.Size([64, 10]) -> that worked!
'''

# Set device
device = torch.device("mps" if torch.has_mps else 'cpu')

# Hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 1 # just 1 pass

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# shuffle: if more than 1 epoch, shuffle it at the end of each epoch

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to mps if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)

        # Get accuracy
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # set all gradients to 0 for each batch
        loss.backward()

        # take gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test of model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
