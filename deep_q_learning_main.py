### Author: Carlos Lassance, Myriam Bontonou, Nicolas Farrugia
### Adaptation by Inès Martinez-Lavayssière

import json
import numpy as np
import time
import random
import pickle
from tqdm import tqdm
from AIs import manh, numpy_rl_reload
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

### The game.py file describes the simulation environment, including the generation of reward and the observation that is fed to the agent.
import game

import rl_cnn as rl


### This set of parameters can be changed in your experiments.
### Definitions :
### - An iteration of training is called an Epoch. It correspond to a full play of a PyRat game.
### - An experience is a set of  vectors < s, a, r, s’ > describing the consequence of being in state s, doing action a, receiving reward r, and ending up in state s'.
###   Look at the file rl.py to see how the experience replay buffer is implemented.
### - A batch is a set of experiences we use for training during one epoch. We draw batches from the experience replay buffer.


epoch = 10000  # Total number of epochs that will be done

max_memory = 1000  # Maximum number of experiences we are storing
number_of_batches = 8  # Number of batches per epoch
batch_size = 32  # Number of experiences we use for training per batch
width = 21  # Size of the playing field
height = 15  # Size of the playing field
cheeses = 40  # Number of cheeses in the game
opponent = manh  # AI used for the opponent

### If load, then the last saved result is loaded and training is continued. Otherwise, training is performed from scratch starting with random parameters.
load = True
save = True


env = game.PyRat()
exp_replay = rl.ExperienceReplay(max_memory=max_memory)

model = rl.ConvNet().double() #model needs to give double parameters, flot64 are rejected


if load:
    model.load()

def play(model, epochs, train=True):

    win_cnt = 0
    lose_cnt = 0
    draw_cnt = 0
    win_hist = []
    cheeses = []
    loss = 0.
    steps = 0.
    last_W = 0
    last_D = 0
    last_L = 0

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for e in tqdm(range(epochs)):
        env.reset()
        game_over = False
        model.eval()
        input_t = torch.DoubleTensor(env.observe()) ######attention
        while not game_over:
            input_tm1 = input_t
#            plt.imshow(input_tm1[0].reshape(29,41))
#            plt.show()
            with torch.no_grad():
                q = model(input_tm1)

                action = torch.argmax(q[0]).item()

            # Apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            input_t = torch.DoubleTensor(input_t)

            # Statistics
            if game_over:
                steps += env.round
                if env.score > env.enemy_score:
                    win_cnt += 1
                elif env.score == env.enemy_score:
                    draw_cnt += 1
                else:
                    lose_cnt += 1
                cheese = env.score
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

        win_hist.append(win_cnt)  # Statistics
        cheeses.append(cheese)  # Statistics

        if train:
            model.train()
            local_loss = 0
            for _ in range(number_of_batches):
                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
                batch_loss = rl.train_on_batch(model, inputs, targets, criterion, optimizer)
                local_loss += batch_loss
            loss += local_loss


        if (e+1) % 100 == 0:  # Statistics every 100 epochs
            cheese_np = np.array(cheeses)
            string = "Epoch {:03d}/{:03d} | Loss {:.4f} | Cheese count {} | Last 100 Cheese {}| W/D/L {}/{}/{} | 100 W/D/L {}/{}/{} | 100 Steps {}".format(
                        e,epochs, loss, cheese_np.sum(),
                        cheese_np[-100:].sum(), win_cnt, draw_cnt, lose_cnt,
                        win_cnt-last_W, draw_cnt-last_D, lose_cnt-last_L, steps/100)
            print(string)
            loss = 0.
            steps = 0.
            last_W = win_cnt
            last_D = draw_cnt
            last_L = lose_cnt

print("Training")
play(model, epoch, True)
if save:
    model.save()
print("Training done")
print("Testing")
play(model, epoch, False)
print("Testing done")
