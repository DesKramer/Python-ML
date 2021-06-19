import numpy as np
import random

def sigmoid(x):
    return 1/(1+np.exp(-x)) 
    
def sigmoid_deriv(x):
    return x*(1-x)

class Agent:
    def __init__(self):
        self.score = 0
        self.turns = 0
        self.rolls = 0
        self.input = np.zeros(5)
        # input represents values for the following: cur_score, points, score_after_turn, rolls_made, enemy_score
        self.syn0 = np.random.rand(self.input.shape[1],6)
        self.syn1 = np.random.rand(6,1)
        self.output = np.zeros(2)
        # output represents probability of doing 1 of 2 things: roll (again), take points and skip
        
    def skip_turn(self):
        self.turns += 1
        
    def roll(self):
        self.turns += 1
        points = random.randint(1,6)
        if (points == 1):
            return 
        self.score += points


# Define method 1 for loss/reward function


if __name__ == '__main__':
    agent1 = Agent()
    agent2 = Agent()


        