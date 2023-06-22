from GoBoard import *
from GoGame import *
import numpy as np

'''
This is a test script for GoBoard
** Read the print instructions at the end of this file! **
Note: 
    1. load_from_numpy() and to_numpy() convert a board to numpy array, 
       where 1 for black stone and -1 for white stone.
    2. valid_moves() returns a list of coordinate.
    3. add_stone() add stones and remove captured stones.
    4. get_scores() return (black_score, white score).
    
    You can replace these functions with your own defined function names,
    or simply use the test data in this script and rewrite other parts.
    
    Please note that the test cases in this script are not sufficient!
'''
board = Board(5) 

# capture test
board.load_from_numpy(np.array([
    [ 0, 0,-1, 0, 0],
    [ 0,-1, 1,-1, 0],
    [-1, 1, 0, 1,-1],
    [ 0,-1, 1,-1, 0],
    [ 0, 0,-1, 0, 0],
]))
# print(board.__str__())
assert set(board.valid_moves( 1))==set([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 4), (3, 0), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)])
board.add_stone(2, 2,-1)
# print(board.__str__())
assert (board.to_numpy()==np.array([
    [ 0, 0,-1, 0, 0],
    [ 0,-1, 0,-1, 0],
    [-1, 0,-1, 0,-1],
    [ 0,-1, 0,-1, 0],
    [ 0, 0,-1, 0, 0],
])).all()
assert set(board.valid_moves( 1))==set([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 4), (3, 0), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)])
assert set(board.valid_moves(-1))==set([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 2), (1, 4), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4), (4, 0), (4, 1), (4, 3), (4, 4)])
board.load_from_numpy(np.array([
    [ 0,-1, 1,-1, 0],
    [-1, 1, 1, 1,-1],
    [ 1, 1, 1, 1, 1],
    [ 1, 1, 1, 1, 0],
    [ 1, 1, 1,-1, 0],
]))
# print(board.__str__())
assert set(board.valid_moves( 1))==set([(0, 0), (0, 4), (3, 4), (4, 4)])
assert set(board.valid_moves(-1))==set([(3, 4), (4, 4)])
board.add_stone(3, 4,-1)
# print(board.__str__())
assert (board.to_numpy()==np.array([
    [ 0,-1, 0,-1, 0],
    [-1, 0, 0, 0,-1],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0,-1],
    [ 0, 0, 0,-1, 0],
])).all()


# ko-rule test
board.load_from_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
]))
assert set(board.valid_moves(-1))==set([(0, 0), (0, 3), (0, 4), (1, 1), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
assert set(board.valid_moves(1)) ==set([(0, 3), (0, 4), (1, 1), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
# print(board.__str__())
board.add_stone(1, 1, 1)
# print(board.__str__())
assert (board.to_numpy()==np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 1, 0, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
])).all()
assert set(board.valid_moves(-1))==set([(0, 0), (0, 3), (0, 4), (1, 4), (2, 0), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
assert set(board.valid_moves(1)) ==set(board.valid_moves(-1))
board.add_stone(1, 4, -1)
board.add_stone(2, 3, 1)
# print(board.__str__())
assert set(board.valid_moves(-1))==set([(0, 0), (0, 3), (0, 4), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)])
assert set(board.valid_moves(1)) ==set(board.valid_moves(-1))
board.add_stone(1, 2, -1)
assert (board.to_numpy()==np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1,-1],
    [ 0,-1, 1, 1, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
])).all()

# score test
board = Board(5) 
board.load_from_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0,-1, 1, 0],
    [ 0,-1, 1, 0, 0],
    [ 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1],
]))
s = board.get_scores()
assert board.get_scores() == [4, 6]
board.add_stone(1, 1, 1)
# print(board.__str__())
assert board.get_scores() == [6, 4]

board.load_from_numpy(np.array([
    [ 0,-1, 1, 0, 0],
    [-1, 0, 0, 1, 0],
    [ 0,-1, 0, 1, 0],
    [ 0,-1, 0, 1, 0],
    [ 0, 0,-1, 0, 1],
]))
# print(board.__str__())
assert board.get_scores() == [10, 10]

board.load_from_numpy(np.array([
    [ 0,-1, 1,-1, 0],
    [-1, 0, 1, 1,-1],
    [ 0, 1, 0, 1, 0],
    [ 1, 0, 1, 1, 0],
    [ 0, 1,-1, 0, 1],
]))
# print(board.__str__())
assert board.get_scores() == [13, 7]
print("TEST SUCCESS! yeah~")
print("Note: the test is not complete, you should also test the following cases:")
print("      1. board hash function")
print("      2. the game end condition")
print("      3. wining condition")
print("      4. other features of GoGame class")
