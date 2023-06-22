from Player import *
from GoGame import GoGame
from tqdm import trange

N_TEST = 80
BOARD_SIZE = 9
N_PLAY_OUT = 50




def single_match(player1, player2, game, display=False):
    player1.player, player2.player = 1, -1
    state = game.reset()
    if display:
        game.display(state)
        print('---------------------------------')
    score = [0, 0, 0]  # player1-win, draw, player2-win

    while True:
        # player 1 move
        action = player1.play(state)
        state = game.next_state(state, 1, action)

        if display:
            game.display(state)
            print('---------------------------------')

        # player 2 move
        if game.is_terminal(state, 1) == 0:  # not end
            action = player2.play(state)
            state = game.next_state(state, -1, action)

        if display:
            game.display(state)
            print('---------------------------------')

        # if end game
        game_end = game.is_terminal(state, -1)  # return -1 for player 1 win
        if game_end != 0:
            score[(int(game_end // 1) + 1)] += 1
            break

    return score


def test_multi_match(player1, player2, game, n_test = 100, print_result=True):
    player1_win, player2_win, draw = 0, 0, 0

    for _ in trange(n_test // 2):
        score = single_match(player1, player2, game)
        player1_win += score[0]
        player2_win += score[2]
        draw += score[1]

    for _ in trange(n_test // 2):  # reverse side
        score = single_match(player2, player1, game)
        player1_win += score[2]
        player2_win += score[0]
        draw += score[1]

    tot_match = (n_test // 2) * 2
    if print_result:
        print("Test result: ")
        print(f"    player1({player1.__class__.__name__})-win:", player1_win, f"{100 * player1_win / tot_match:.2f}%")
        print(f"    player2({player2.__class__.__name__})-win:", player2_win, f"{100 * player2_win / tot_match:.2f}%")
        print("    draw:", draw, f"{100 * draw / tot_match:.2f}%")
    return player1_win, player2_win, draw

if __name__ == '__main__':
    # ** Modify this part to test different players **
    global_game = GoGame(n=BOARD_SIZE)
    random_player = RandomPlayer(global_game, 1)
    alphazero_player = AlphaZeroPlayer(global_game, "./temp/best.pth", 50, 1.0)
    
    player1 = alphazero_player
    player2 = random_player

    # test win-lose rate
    test_multi_match(player1, player2, global_game, N_TEST)

    # visualize games
    # player1 play first
    # single_match(player1, player2, game, display=True)
    # player2 play first
    # single_match(player2, player1, game, display=True)

    # play with human
    # player1 = alphabeta_player
    # player2 = human_player
    # single_match(player1, player2, game, display=True)
