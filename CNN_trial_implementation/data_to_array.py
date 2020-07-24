#import data
################################################@

import scipy
import scipy.sparse
import ast
import numpy as np
import os
import tqdm

PHRASES = {
    "# Random seed\n": "seed",
    "# MazeMap\n": "maze",
    "# Pieces of cheese\n": "pieces"    ,
    "# Rat initial location\n": "rat"    ,
    "# Python initial location\n": "python"   ,
    "rat_location then python_location then pieces_of_cheese then rat_decision then python_decision\n": "play"
}

MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

translate_action = {
    MOVE_LEFT:0,
    MOVE_RIGHT:1,
    MOVE_UP:2,
    MOVE_DOWN:3
}# This data structures defines the encoding of the four possible movements

def process_file_2(filename):
    f = open(filename,"r")
    info = f.readline()
    params = dict(play=list())
    while info is not None:
        if info.startswith("{"):
            params["end"] = ast.literal_eval(info)
            break
        if "turn " in info:
            info = info[info.find('rat_location'):]
        if info in PHRASES.keys():
            param = PHRASES[info]
            if param == "play":
                rat = ast.literal_eval(f.readline())
                python = ast.literal_eval(f.readline())
                pieces = ast.literal_eval(f.readline())
                rat_decision = f.readline().replace("\n","")
                python_decision = f.readline().replace("\n","")
                play_dict = dict(
                    rat=rat,python=python,piecesOfCheese=pieces,
                    rat_decision=rat_decision,python_decision=python_decision)
                params[param].append(play_dict)
            else:
                params[param] = ast.literal_eval(f.readline())
        else:
            print("did not understand:", info)
            break
        info = f.readline()
    return params

def dict_to_x_y(end,rat, python, maze, piecesOfCheese,rat_decision,python_decision,
                mazeWidth=21, mazeHeight=15):
    # We only use the winner
    if end["win_python"] == 1:
        player = python
        opponent = rat
        decision = python_decision
    elif end["win_rat"] == 1:
        player = rat
        opponent = python
        decision = rat_decision
    else:
        return False
    if decision == "None" or decision == "": #No play
        return False
    x_1 = convert_input_2(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese)
    y = np.zeros((1,4),dtype=np.int8)
    y[0][translate_action[decision]] = 1
    return x_1,y

### The goal of this function is to create a canvas, which will be the vector used to train the classifier.
### As we want to predict a next move, we will create a canvas that is centered on the player, so that we can easily with the translation invariance.


def convert_input_2(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
	# We will consider twice the size of the maze to simplify the creation of the canvas
	# The canvas is initialized as a numpy tensor with 3 modes (meaning it is indexed using three integers), the third one corresponding to "layers" of the canvas.
	# Here, we just use one layer, but you can defined other ones to put more information on the play (e.g. the location of the opponent could be put in a second layer)

    im_size = (1,2*mazeHeight-1,2*mazeWidth-1)

    # We initialize a canvas with only zeros
    canvas = np.zeros(im_size)


    (x,y) = player

    # fill in the first layer of the canvas with the value 1 at the location of the cheeses, relative to the position of the player (i.e. the canvas is centered on the player location)
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[0,y_cheese+center_y-y,x_cheese+center_x-x] = 1
    return canvas

games = list()
directory = "/Users/IML/Dossier/IMT_Atlantique/3A_bis/IA-introduction/PyRat-master/saves/"

for root, dirs, files in os.walk(directory):
    for filename in tqdm.tqdm(files):
        if filename.startswith("."):
            continue
        game_params = process_file_2(directory+filename)
        games.append(game_params)

x_1_train = list()
y_train = list()
wins_python = 0
wins_rat = 0
for game in tqdm.tqdm(games):
    if game["end"]["win_python"] == 1:
        wins_python += 1
    elif game["end"]["win_rat"] == 1:
        wins_rat += 1
    else:
        continue
    plays = game["play"]
    for play in plays:
        x_y = dict_to_x_y(**play,maze=game_params["maze"],end=game["end"])
        if x_y:
            x1, y = x_y
            y_train.append(scipy.sparse.csr_matrix(y.reshape(1,-1)))
            #x_1_train.append(scipy.sparse.csr_matrix(x1.reshape(1,-1)))
            x_1_train.append(x1)
print("Greedy/Draw/Greedy, {}/{}/{}".format(wins_rat,1000 - wins_python - wins_rat, wins_python))

np.savez_compressed("dataset.npz",x=x_1_train,y=y_train)

