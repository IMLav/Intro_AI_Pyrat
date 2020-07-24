
################################################@
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle, scipy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 5
learning_rate = 0.001
number_of_batches = 6  # Number of batches per epoch

mazeWidth = 21
mazeHeight = 15


class PyratDataset(Dataset):

    def __init__(self):
    #data loading
      x = np.load("dataset.npz",allow_pickle = True)['x']
      print(x.shape)
      print(x[0,:,:,:].shape)
      self.x = x
      print(self.x.shape)
      y = np.load("dataset.npz",allow_pickle = True)['y']
      y = scipy.sparse.vstack(y).todense()
      self.y = np.argmax(np.array(y),1)
      self.n_sample = y.shape[0]


    def __getitem__(self, index):

      return self.x[index], self.y[index]

    def __len__(self):
        return self.n_sample


dataset = PyratDataset()
train_dataset, test_dataset = train_test_split(dataset, test_size=0.20, random_state=1)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape, labels.shape)

classes = ('left', 'right', 'up', 'down')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool1= nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 16, 5)
        self.fc1 = nn.Linear(16 * 8 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)


    def forward(self, x):


          x = self.pool1(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
          x = F.relu(self.conv2(x)) # -> n, 16, 5, 5
          x = x.view(-1, 16 * 8 * 14)            # -> n, 400
          x = F.relu(self.fc1(x))               # -> n, 120
          x = F.relu(self.fc2(x))               # -> n, 84
          x = self.fc3(x)                       # -> n, 10
          return x


model = ConvNet().to(device)
model.double() #otherwise the parameters of the model are in float64 and are not accepted


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 1 input channels, 6 output channels, 5 kernel size

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
           # print(i)
            #print(labels.shape)

            label = labels[i]
            #print(label)
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')


PATH = 'save_rl/cnn.pth'
torch.save(model.state_dict(), PATH)
