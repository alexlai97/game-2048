import numpy as np
import random
from typing import Tuple, List, Optional


class Game2048:
    def __init__(self):
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.moved = False
        self._spawn_tile()
        self._spawn_tile()
    
    def _spawn_tile(self) -> None:
        empty_cells = [(i, j) for i in range(4) for j in range(4) if self.grid[i, j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i, j] = 2 if random.random() < 0.9 else 4
    
    def _slide_line(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        line = line[line != 0]
        score_gained = 0
        
        i = 0
        while i < len(line) - 1:
            if line[i] == line[i + 1]:
                line[i] *= 2
                score_gained += line[i]
                line = np.delete(line, i + 1)
                self.moved = True
            i += 1
        
        result = np.zeros(4, dtype=int)
        result[:len(line)] = line
        return result, score_gained
    
    def move_left(self) -> bool:
        self.moved = False
        total_score = 0
        
        for i in range(4):
            original_row = self.grid[i].copy()
            self.grid[i], score = self._slide_line(self.grid[i])
            total_score += score
            
            if not np.array_equal(original_row, self.grid[i]):
                self.moved = True
        
        self.score += total_score
        return self.moved
    
    def move_right(self) -> bool:
        self.grid = np.fliplr(self.grid)
        moved = self.move_left()
        self.grid = np.fliplr(self.grid)
        return moved
    
    def move_up(self) -> bool:
        self.grid = self.grid.T
        moved = self.move_left()
        self.grid = self.grid.T
        return moved
    
    def move_down(self) -> bool:
        self.grid = self.grid.T
        moved = self.move_right()
        self.grid = self.grid.T
        return moved
    
    def move(self, direction: str) -> bool:
        moves = {
            'left': self.move_left,
            'right': self.move_right,
            'up': self.move_up,
            'down': self.move_down
        }
        
        if direction not in moves:
            return False
        
        if moves[direction]():
            self._spawn_tile()
            return True
        return False
    
    def get_available_moves(self) -> List[str]:
        moves = []
        for direction in ['left', 'right', 'up', 'down']:
            test_game = Game2048()
            test_game.grid = self.grid.copy()
            test_game.score = self.score
            if getattr(test_game, f'move_{direction}')():
                moves.append(direction)
        return moves
    
    def is_game_over(self) -> bool:
        return len(self.get_available_moves()) == 0
    
    def has_won(self) -> bool:
        return np.max(self.grid) >= 2048
    
    def get_state(self) -> str:
        if self.has_won():
            return "won"
        elif self.is_game_over():
            return "lost"
        else:
            return "ongoing"
    
    def get_grid_copy(self) -> np.ndarray:
        return self.grid.copy()
    
    def __str__(self) -> str:
        result = f"Score: {self.score}\n"
        result += "+------+------+------+------+\n"
        for row in self.grid:
            result += "|"
            for cell in row:
                if cell == 0:
                    result += "      |"
                else:
                    result += f" {cell:4d} |"
            result += "\n+------+------+------+------+\n"
        return result