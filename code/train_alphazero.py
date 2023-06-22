import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import datetime

import numpy as np
from tqdm import tqdm

from pit import test_multi_match
from multipit import mul_test_multi_match
from Player import FastEvalPlayer
from Player import RandomPlayer
from MCTS import MCTS
from GoGame import GoGame

import torch.multiprocessing as mp
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def static_collect_single_game(board_size, nnet, num_sims, cpuct):
    game = GoGame(board_size)
    mcts = MCTS(game, nnet, num_sims, cpuct)
    mcts.train()
    game_history = []
    board = game.reset()
    current_player = 1
    current_step = 0
    # self-play until the game is ended
    while True:
        current_step += 1
        pi = mcts.get_action_prob(board, current_player)
        data = game.get_transform_data(game.get_board(board, current_player), pi)
        for b, p in data:
            game_history.append([b, current_player, p, None])
        action = np.random.choice(len(pi), p=pi)
        board = game.next_state(board, current_player, action)
        current_player *= -1
        game_result = game.is_terminal(board, current_player)
        if game_result != 0:  # Game Ended
            return [(x[0], x[2], game_result * ((-1) ** (x[1] != current_player))) for x in game_history]

class Trainer():
    """
    """

    def __init__(self, game: GoGame, nnet, config):
        self.game = game
        self.next_net = nnet
        self.last_net = self.next_net.__class__(self.game)
        self.config = config
        self.mcts = MCTS(self.game, self.next_net, self.config["num_sims"], self.config["cpuct"])
        self.train_data_packs = []


    def collect_single_game(self):
        """
        Collect self-play data for one game.
        
        @return game_history: A list of (board, pi, z)
        """
        # create a New MCTS 
        self.mcts = MCTS(self.game, self.next_net, self.config["num_sims"], self.config["cpuct"])
        self.mcts.train()

        game_history = []
        board = self.game.reset()
        current_player = 1
        current_step = 0

        # self-play until the game is ended
        while True:
            current_step += 1
            pi = self.mcts.get_action_prob(board, current_player)
            data = self.game.get_transform_data(self.game.get_board(board, current_player), pi)
            for b, p in data:
                game_history.append([b, current_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board = self.game.next_state(board, current_player, action)
            current_player *= -1
            game_result = self.game.is_terminal(board, current_player)

            if game_result != 0:  # Game Ended
                return [(x[0], x[2], game_result * ((-1) ** (x[1] != current_player))) for x in game_history]

    def train(self):
        """
        Main Training Loop of AlphaZero
        each iteration:
            * Collect data by self play
            * Train the network
            * Pit the new model against the old model
                If the new model wins, save the new model, and evaluate the new model
                Otherwise, reject the new model and keep the old model
            
        """
        losses = []
        iters = []
        f_iter = []
        update = []
        for i in range(1, self.config["max_training_iter"] + 1):

            log.info(f'Starting Iter #{i} ...')

            data_pack = deque([])
            cpuct:float = self.config["cpuct"]
            T = tqdm(range(self.config["selfplay_each_iter"]), desc="Self Play")
            if not self.config["multiprocessing"]:
                
                for _ in T:
                    game_data = self.collect_single_game()
                    data_pack += game_data
                    r = game_data[0][-1]
                    T.set_description_str(f"Self Play win={r}, len={len(game_data)}")
            else:
                self.next_net.nnet.share_memory()
                mp.set_start_method("spawn", force=True)
                num_sims:int = self.config["num_sims"]
                cpuct:float = self.config["cpuct"]
                with mp.Pool(processes=10) as pool:
                    for game_data in pool.starmap(func=static_collect_single_game, 
                                                  iterable=[
                        (self.game.n, self.next_net, num_sims, cpuct,)
                    ] * self.config["selfplay_each_iter"],):
                        data_pack += game_data
                        r = game_data[0][-1]
                        T.set_description_str(f"Self Play win={r}, len={len(game_data)}")
            self.train_data_packs.append(data_pack)

            if len(self.train_data_packs) > self.config["max_train_data_packs_len"]:
                log.warning(
                    f"Removing the oldest data pack...")
                self.train_data_packs.pop(0)

            trainExamples = []
            for e in self.train_data_packs:
                trainExamples.extend(e)
            shuffle(trainExamples)

            self.next_net.save_checkpoint(folder=self.config["checkpoint_folder"], filename='temp.pth.tar')
            self.last_net.load_checkpoint(folder=self.config["checkpoint_folder"], filename='temp.pth.tar')

            losses += self.next_net.train(trainExamples)
            iters += [i+0.1*j for j in range(10)]
            
            
            # import ipdb; ipdb.set_trace()
            next_mcts = MCTS(self.game, self.next_net, self.config["num_sims"], self.config["cpupt"])
            last_mcts = MCTS(self.game, self.last_net, self.config["num_sims"], self.config["cpupt"])

            log.info('Pitting against last version...')
            ######################################
            #        YOUR CODE GOES HERE         #
            ###################################### 
            # Pitting against last version, and decide whether to save the new model
            next_mcts_player, last_mcts_player = FastEvalPlayer(next_mcts), FastEvalPlayer(last_mcts)
            random_player = RandomPlayer(self.game, 1)
            multi_match = (mul_test_multi_match if self.config["multiprocessing"] else test_multi_match)
            next_wins, next_loses, next_draws = multi_match(next_mcts_player, random_player, self.game)
            last_wins, last_loses, last_draws = multi_match(last_mcts_player, random_player, self.game)
            f_iter.append(i)
            if next_wins - last_wins <= 1:
                log.info("reject new model")
                update.append(0)
                self.next_net.load_checkpoint(folder=self.config["checkpoint_folder"], filename='temp.pth.tar')
            else:
                update.append(1)
                log.info("accept new model")
                self.next_net.save_checkpoint(folder=self.config["checkpoint_folder"], filename='checkpoint_'+str(i)+'.pth.tar')
                self.next_net.save_checkpoint(folder=self.config["checkpoint_folder"], filename='best.pth.tar')
            plt.figure(figsize=(15, 6))
            plt.subplot(1, 2, 1)
            plt.title("loss")
            plt.plot(iters, losses)
            plt.xlabel('iter (10 step per iter)')
            plt.ylabel('loss')
            plt.subplot(1, 2, 2)
            plt.title("update rate")
            plt.plot(f_iter, update)
            plt.xlabel('iter')
            plt.ylabel('update')
            plt.suptitle("Training curve", fontsize=16)
            plt.savefig('result_2.png')
            plt.close()