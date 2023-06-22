import numpy as np
from GoBoard import Board


class GoGame:
    def __init__(self, n=3):
        assert n % 2 == 1
        self.n = n
        self.vir = 0

    def reset(self):
        """
        Reset the game.
        """
        return Board(self.n)
    
    def obs_size(self):
        """
        Size of the board.
        """
        return self.n, self.n

    def action_size(self):
        """
        Number of all possible actions.
        """
        return self.n * self.n + 1 # the extra 1 is for 'pass'(虚着), a action for doing nothing

    def get_board(self, board: Board, player: int) -> Board:
        """
        Convert given board to a board from player1's perspective.
        If current player is player1, do nothing, else, reverse the board.
        This can help you write neater code for search algorithms as you can go for the maximum return every step.

        @param board: the board to convert
        @param player: 1 or -1, the player of current board

        @return: a board from player1's perspective
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        # import ipdb; ipdb.set_trace()
        # print(board)
        tboard = board.copy()
        tboard.data *= player
        # tboard.vir = board.vir
        return tboard
        ######################################

    def next_state(self, board: Board, player: int, action: int):
        """
        Get the next state by executing the action for player.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        # Note: don't forget 'pass'
        # pass
        tboard = board.copy()
        if action == self.n * self.n:
            tboard.vir = tboard.vir + 1
            return tboard
        tboard.vir = 0
        # print(board.moves)
        
        row, col = action//self.n, action%self.n
        next_board = tboard.copy()
        next_board.add_stone(row, col, player)
        return next_board

    def get_transform_data(self, board, pi):
        # import ipdb; ipdb.set_trace()
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        tboard = board.copy()
        lt = []
        for i in range(1, 5):
            for j in [True, False]:
                # import ipdb; ipdb.set_trace()
                new_board = np.rot90(tboard.data, i)
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_board = np.fliplr(new_board)
                    new_pi = np.fliplr(new_pi)
                lt += [(new_board, list(new_pi.ravel()) + [pi[-1]])]
        return lt

    def get_valid_moves(self, board: Board, player: int):
        """
        Get a binary vector of length self.action_size(), 1 for all valid moves.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        valid_moves = np.zeros(self.action_size())
        tboard = board.copy()
        for item in range(self.action_size()):
            valid_moves[item] = tboard.check_valid_move(item//self.n, item%self.n, player)
        valid_moves[-1] = True
        return valid_moves
        ######################################

    # @staticmethod
    def is_terminal(self, board: Board, player: int):
        """
        Check whether the game is over.
        @return: 1 or -1 if player or opponent wins, 1e-4 if draw, 0 if not over
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        tboard = board.copy()
        if tboard.is_game_over():
            scores = tboard.get_scores()
            if scores[0] == scores[1]:
                return 1e-4
            else:
                if player == 1:
                    return 1 if scores[0] > scores[1] else -1
                else:
                    return 1 if scores[1] > scores[0] else -1
        # Note: end game when two consecutive passes or too many moves, compare scores of two players.
        return 0

    @staticmethod
    def get_string(board: Board):
        """
        Convert the board to a string.
        """
        ######################################
        #        YOUR CODE GOES HERE         #
        # return np.array2string(board.data)
        # import ipdb; ipdb.set_trace()
        tboard = board.copy()
        return tboard.tostring()
        ######################################
        # Note: different board(Game Statue) must return different string
        # return None

    @staticmethod
    def display(board: Board):
        """
        Print the board to console.
        """
        print(str(board))
        
    
