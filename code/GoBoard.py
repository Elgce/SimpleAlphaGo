import numpy as np


class Stone:
    EMPTY = 0
    BLACK = 1
    WHITE = -1


class Board:
    def __init__(self, n):
        assert n % 2 == 1
        self.n = n
        self.data = np.zeros((n, n))
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.num_stones = 0
        self.last_move = {}
        self.vir = 0

    def __str__(self) -> str:
        ret = ''
        for i in range(self.n):
            for j in range(self.n):
                if self.data[i, j] == Stone.EMPTY:
                    ret += '+'
                elif self.data[i, j] == Stone.BLACK:
                    ret += '○'
                else:
                    ret += '●'
            ret += '\n'
        return ret

    def copy(self):
        board = Board(self.n)
        board.data = self.data.copy()
        # board.data = self.data
        board.num_stones = self.num_stones  
        board.last_move = self.last_move
        board.vir = self.vir
        return board
    
    def load_from_numpy(self, a: np.ndarray):
        assert a.shape == (self.n, self.n)
        self.data = a

    def to_numpy(self):
        return self.data.copy()
        # Note: Copy if you don't want to mess up the original board.

    def is_valid_position(self, x, y):
        # check if (x, y) is a valid position on the board
        return x >= 0 and x < self.n and y >=0 and y < self.n
    
    def add_stone(self, x, y, color):
        '''
        Add a stone to the board, and remove captured stones
        '''
        ######################################
        #        YOUR CODE GOES HERE         #
        # import ipdb; ipdb.set_trace()
        assert self.is_valid_position(x, y) and self.data[x ,y] == Stone.EMPTY
        self.data[x, y] = color
        self.remove_captured_stones(x, y, color)
        self.num_stones += 1
        ######################################
    
    def get_group_liberty(self, x, y):
        '''
        accoarding to (x, y), get all linked pos. check if there are any literties
        input: x, y
        output: group, liberty
        description:
            group: [], all linked pos, including (x, y) itself
            liberty: int, num of liberties this group have
        '''
        color = self.data[x, y]
        liberty = set()
        group = set()
        assert color != Stone.EMPTY
        def dfs(x, y):
            if (x, y) in group:
                return
            group.add((x ,y))
            for dx, dy in self.moves:
                nx, ny = x + dx, y + dy
                if self.is_valid_position(nx, ny):
                    if self.data[nx, ny] == Stone.EMPTY:
                        liberty.add((nx, ny))
                    elif self.data[nx, ny] == color:
                        dfs(nx, ny)
        dfs(x, y)
        num_liberty = len(liberty)
        return group, num_liberty
    
    def remove_captured_stones(self, x, y, color):
        opp_color = -color
        # find all dead ones and remove them together, otherwise, fail
        captured_stones = []
        for i, j in self.moves:
            nx, ny = x + i, y + j
            if self.is_valid_position(nx, ny) and self.data[nx, ny] == opp_color:
                group, liberty = self.get_group_liberty(nx, ny)
                if liberty == 0:
                    captured_stones.extend(group)
        for item in captured_stones:
            self.data[item] = Stone.EMPTY
        self.last_move['color'] = color
        self.last_move['captured_stones'] = captured_stones
                    
        
    def undo_add_stone(self, i, j):
        color = self.last_move.get('color')
        captured_stones = self.last_move.get('captured_stones')
        self.data[i, j] = Stone.EMPTY
        for x, y in captured_stones:
            self.data[x, y] = -color
        self.num_stones -= 1
        
    def check_valid_move(self, i, j, color):
        if not self.is_valid_position(i, j) or self.data[i, j] != Stone.EMPTY:
            return False
        data = self.data.copy()
        last_move = self.last_move.copy()
        self.data[i, j] = color
        self.remove_captured_stones(i, j, color)
        group, liberty = self.get_group_liberty(i, j)
        self.last_move = last_move.copy()
        self.data = data.copy()
        if liberty > 0:
            last_capture = self.last_move.get('captured_stones')
            last_move = self.last_move.copy()
            if last_capture is not None and len(last_capture) == 1:
                if (i, j) in last_capture:
                    self.add_stone(i, j, color)
                    capture = self.last_move.get('captured_stones')
                    if capture is not None and len(capture) == 1:
                        self.undo_add_stone(i, j)
                        self.last_move = last_move.copy()
                        return False        
        return liberty > 0
        # liberty condition
        
    def valid_moves(self, color):
        '''
        Return a list of avaliable moves
        @return: a list like [(0,0), (0,1), ...]
        '''
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        valid_moves = []
        for i in range(self.n):
            for j in range(self.n):
                if self.data[i, j] == Stone.EMPTY and self.check_valid_move(i, j, color):
                    valid_moves.append((i, j))
        return valid_moves
    
    def is_game_over(self):
        return (len(self.valid_moves(Stone.BLACK)) == 0 and len(self.valid_moves(Stone.WHITE)) == 0)\
            or (self.num_stones > self.n * self.n *4) or self.vir >= 2

    def tostring(self):
        # import ipdb; ipdb.set_trace()
        # print(self.data.shape)
        tdata = self.data.copy()
        return np.array2string(tdata.astype(int))

    def reset_vir(self):
        self.vir = 0
    
    def add_vir(self):
        self.vir += 1

    def get_scores(self):
        '''
        Compute score of players
        @return: a tuple (black_score, white_score)
        '''
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        black_score = np.sum(self.data == Stone.BLACK)
        white_score = np.sum(self.data == Stone.WHITE)
        # compute scores from empty position
        black_group = set()
        white_group = set()
        empty_group = set()
        def dfs(i, j):
            assert self.is_valid_position(i, j)
            if visited[i, j] == 1:
                return
            visited[i, j] = 1
            empty_group.add((i, j))
            for dx, dy in self.moves:
                nx, ny = i + dx, j + dy
                if self.is_valid_position(nx, ny):
                    if self.data[nx, ny] == Stone.BLACK:
                        black_group.add((nx, ny))
                    elif self.data[nx, ny] == Stone.WHITE:
                        white_group.add((nx, ny))
                    else:
                        dfs(nx, ny)
            
        visited = np.zeros(shape=(self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if self.data[i, j] == Stone.EMPTY and visited[i, j] == 0:
                    dfs(i, j)
                    blacks, whites, emptys= len(black_group), len(white_group), len(empty_group)
                    white_score += emptys if blacks == 0 else 0
                    black_score += emptys if whites == 0 else 0
                    empty_group.clear()
                    black_group.clear()
                    white_group.clear()
        return [black_score, white_score]
