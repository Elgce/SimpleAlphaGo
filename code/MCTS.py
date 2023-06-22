import math
import pickle
import os

import numpy as np
from util import *
from GoNNet import GoNNetWrapper
from GoGame import GoGame
from GoBoard import Board

class MCTS:
    def __init__(self, game: GoGame, nnet: GoNNetWrapper, num_sims, C):
        self.game = game
        self.num_sims = num_sims
        self.nnet = nnet
        self.C = C
        self.training = True
        # stores Q=U/N values for (state,action)
        self.Q_state_action = {}
        # stores times edge (state,action) was visited
        self.N_state_action = {}
        # stores times board state was visited
        self.N_state = {}
        # stores policy
        self.P_state = {}

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

                          
    def get_action_prob(self, board, player):
        board = self.game.get_board(board, player)
        if self.training:
            for i in range(self.num_sims):
                self.search(board)

        s = self.game.get_string(board)
        possible_actions = self.get_possible_actions(board)
        counts = np.array([self.N_state_action[(s, a)] if (s, a) in self.N_state_action and a in possible_actions else 0 for a in range(self.game.action_size())])
        sum_count = counts.sum()
        if sum_count:
            probs = counts / sum_count
        else:
            probs = np.ones(len(counts), dtype=float)/len(counts)
        return probs                                                      

    
    def get_possible_actions(self, board):
        tboard = board.copy()
        return np.where(self.game.get_valid_moves(tboard, 1) == 1)[0]

    def search(self, board: Board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound.
        """
        # use the current board state to get a unique string representation as a key
        s = self.game.get_string(board)
        # print(s)
        # TODO handles leaf node
        ##############################
        # YOUR CODE GOES HERE
        terminal = self.game.is_terminal(board, 1)
        if terminal != 0:
            return -terminal if terminal in [-1, 1] else 0
        if s not in self.P_state:
            self.P_state[s], v = self.nnet.predict(board.data)
            valids = self.game.get_valid_moves(board, 1)
            self.P_state[s] = self.P_state[s] * valids
            sum_P_state = np.sum(self.P_state[s])
            if sum_P_state > 0:
                self.P_state[s] /= sum_P_state
            else:
                self.P_state[s] = self.P_state[s] + valids
                self.P_state[s] /= np.sum(self.P_state[s])  
            self.N_state[s] = 0
            return -v
        highest_ucb = -float('inf')
        best_act = -1
        possible_actions = self.get_possible_actions(board)
        for act in possible_actions:
            if (s, act) in self.Q_state_action:
                ucb = self.Q_state_action[(s, act)] + self.C * self.P_state[s][act] * math.sqrt(self.N_state[s]) / (1 + self.N_state_action[(s, act)])
            else:
                ucb = self.C * self.P_state[s][act] * math.sqrt(self.N_state[s] + EPS)
            if ucb > highest_ucb:
                highest_ucb = ucb
                best_act = act
        action = best_act
        next_s = self.game.next_state(board, 1, action)
        next_s = self.game.get_board(next_s, -1)

        v = self.search(next_s)

        if (s, action) in self.Q_state_action:
            self.Q_state_action[(s, action)] = (self.N_state_action[(s, action)] * self.Q_state_action[(s, action)] + v) / (self.N_state_action[(s, action)] + 1)
            self.N_state_action[(s, action)] += 1
        else:
            self.Q_state_action[(s, action)] = v
            self.N_state_action[(s, action)] = 1
        self.N_state[s] += 1
        return -v

    def save_params(self, file_name="mcts_param.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load_params(self, file_name="mcts_param.pkl"):
        if not os.path.exists(file_name):
            print(f"Parameter file {file_name} does not exist, load failed!")
            return False
        with open(file_name, "rb") as f:
            self.__dict__ = pickle.load(f)
            print(f"Loaded parameters from {file_name}")
        return True