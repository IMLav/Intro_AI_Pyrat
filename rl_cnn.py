### Author: Carlos Lassance, Myriam Bontonou, Nicolas Farrugia
### Adaptation by Ines Martinez-Lavayssiere

##################################################################
#imports


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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

PATH = 'cnn.pth'

## Part 1 - Model to learn the Q-function
##################################################################
# Creating a Convolutional Neural Network, inspired by Python Engineer, https://www.youtube.com/channel/UCbXgNpp0jedKWcQiULLbDTA

class ConvNet(nn.Module):

    def __init__(self, batch_size = 5):
        super(ConvNet, self).__init__()

        #setting the batch_size
        self.batch_size = batch_size


        #creating the architecture of the CNN
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.pool1= nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, 16, 5)
        self.fc1 = nn.Linear(16 * 8 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4) #outcode = 4, as the number of possible labels

    def forward(self, x):

          x = self.pool1(F.relu(self.conv1(x)))
          x = F.relu(self.conv2(x))
          x = x.view(-1, 16 * 8 * 14)            #flattening the vector, hence 16 * 8 * 14 which is the current dimension of x
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x


    def load(self):
        self.load_state_dict(torch.load(PATH))

    def save(self):
        torch.save(self.state_dict(), PATH)



#Training on batches to perform the reinforcement learning and pursue the models training
def train_on_batch(model, inputs, targets, criterion, optimizer):
    ### Simple helper function to train the model given a batch of inputs and targets, optimizes the model and returns the loss
    #targets = torch.FloatTensor(targets)
    targets = targets.type(torch.DoubleTensor)
    inputs = inputs.type(torch.DoubleTensor)


    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    return loss.item()



## Part 2 - Experience Replay
##################################################################
#From TP4 correction

class ExperienceReplay(object):

    #During gameplay all experiences < s, a, r, s’ > are stored in a replay memory.
    #During training, batches of randomly drawn experiences are used to generate the input and target for training.

    def __init__(self, max_memory=100, discount=.9):

        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, experience, game_over):
        # Save an experience to memory
        self.memory.append([experience, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        # How many experiences do we have?
        len_memory = len(self.memory)

        # Number of actions that can possibly be taken in the game (up, down, left, right)
        num_actions = 4

        # Dimensions of the game field
        env_dim = list(self.memory[0][0][0].shape)
        env_dim[0] = min(len_memory, batch_size)


        # We want to return an input and target vector with inputs from an observed state...
        inputs = torch.zeros(env_dim)
        #...and the target r + gamma * max Q(s’,a’)
        # Note that our target is a matrix, with possible fields not only for the action taken but also
        # for the other possible actions. The actions do not take the same values as the prediction to not affect them.
        Q = torch.zeros((inputs.shape[0], num_actions))

        # We randomly draw experiences to learn from
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):

#            idx = -1
            state, action_t, reward_t, state_tp1 = self.memory[idx][0]
            # We also need to know whether the game ended at this state
            game_over = self.memory[idx][1]

            # Add the state s to the input
            inputs[i:i+1] = state
            # First, we fill the target values with the predictions of the model.
            # They will not be affected by training (since the training loss for them is 0).
            model.eval()
            with torch.no_grad():
                Q[i] = model(state)[0]

                # If the game ended, the expected reward Q(s,a) should be the final reward r.
                # Otherwise the target value is r + gamma * max Q(s’,a’)

                # If the game ended, the reward is the final reward
                if game_over:  # if game_over is True
                    Q[i, action_t] = reward_t
                else:
                    # r + gamma * max Q(s’,a’)
                    next_round = model(state_tp1)[0]
                    Q[i, action_t] = reward_t + self.discount * torch.max(next_round)
        return inputs, Q

    def load(self):
        self.memory = pickle.load(open("save_rl/memory.pkl","rb"))

    def save(self):
        pickle.dump(self.memory,open("save_rl/memory.pkl","wb"))
