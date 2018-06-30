# Deep-Q-Learning
Implementations of deep q-networn, dueling q-network with base or double q-learning training algorithm, tested on OpenAI Gym. 

## Prerequisites

The project is implemented using Python 3.5 and Tensorflow (tested with tensorflow-gpu 1.2.1).
The usable environments are from OpenAi Gym. For installing gym look at https://github.com/openai/gym 
To work need atari_wrappers.py from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

##  Train a network

Networks training is performed using the train.py module. It requires as parameter the gym environment to be learned. Optionally it is possible to specify the type of network and learning algorithm to be used. Training can restart from a checkpoint using the --checkpoint argument to provide the network weights and --training_info argument to provide the training status (es. current step, total steps, experience replay buffer data). By using --checkpoint_step it is possible to specify after how many steps a checkpoint is saved.

```
>python train.py --environment PongNoFrameskip-v0 --network doubledqn --checkpoint path/to/weights --training_info path/to/training_info
```

## Test a network

The developed models can be tested using the test.py module. It requires the environment and the model weights as arguments (weights must correspond to the specified network type). It is possible to specify the epsilon-greedy value that determines the ratio between network action and random actions.  The argument --show will render the game. 

```
>python test.py --environment PongNoFrameskip-v0 --network doubledqn --model path/to/weights --epsilon 0.05 --num_games 100 --show
```

The output will be a the average collected score and the max score. Moreover it will be saved a file containing all the rewards collected during the test. Use PickleSerializer and GameStats in util.py to read the content.

```
...
gs = util.GameStats()
util.PickleSerializer.load(gs, "path_to_file")
print(gs.games_rewards)
...
```


## References
The work is based on the following list of articles:

* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

## Authors

**Ivan Bergonzani**

