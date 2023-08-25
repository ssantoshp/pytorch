import torch
import torch.nn as nn # to build our neural net + loss funcs
import torch.optim as optim # optimization alog like adams, gradient descent
import torch.nn.functional as F # stuff like relu, func that do not have params
from torch.utils.data import DataLoader # helps us create mini batches on data
import torchvision.datasets as datasets # pytorch built in datasets
import torchvision as transforms # transformation we can do on our dataset

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

'''
model = NN(784, 10) # 28*28 = 784, 10 digits/classes
x = torch.randn(64, 784) # 64 examples
print(model(x).shape) # we want output to be 64x10, prob of each classes for each example
# out: torch.Size([64, 10]) -> that worked!
'''

# Set device
device = torch.device("mps" if torch.has_mps else 'cpu')


# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1 # just 1 pass

# Load Data
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# shuffle: if more than 1 epoch, shuffle it at the end of each epoch

test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

