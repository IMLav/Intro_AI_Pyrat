# Intro_AI_Pyrat

This is my implementation of an AI designed to play the game Pyrat, a game developped by IMT Atlantique.

To find out how this game works, please visit this page :https://formations.imt-atlantique.fr/pyrat/pyrat-doc/

Please note that this website describes an introductory course from IMT Atlantique. This repository regards a more advanced class building on the game presented. 
Hence, only the game described is relevent to this repository.

# Relevent files 

This repository contains both the game files as well as the ressources I developped for my AI. 
While the authors of the code are stated at the head of each file, here is the list of the code I adapted and whish to put forward : 

- AIs/deep_q_learning_AI.py
- deep_q_learning_main.py
- rl_cnn.py

I also included in this repository an implementation of a CNN, which can be found in CNN/, I developped the following files.

- cnn_model.py
- data_to_array.py

# How to launch the game using the AI I developped

## Installing the pyrat Game

Please follow the necessary steps presented in this guide :
https://formations.imt-atlantique.fr/pyrat/install/

You shouldn't need to download the game from the repository provided in the tutorial : it is already included in the present repository.

## Lauching the pyrat Game using deep_q_learning_AI.py using the correct game parameters
MacOS : 
python pyrat.py -p 40 -md 0 -d 0 --nonsymmetric --rat AIs/deep_q_learning_AI.py --python AIs/manh.py

Linux :
python3 pyrat.py -p 40 -md 0 -d 0 --nonsymmetric --rat AIs/deep_q_learning_AI.py --python AIs/manh.py

# Licence

As per the LICENCE file provided at the root of the project, this code is protected by the GNU GENERAL PUBLIC LICENSE. You can republish this code but cannot alter it or make its license more restrictive.


