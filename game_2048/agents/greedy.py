#!/usr/bin/env python3

import numpy as np
from typing import Dict, Tuple
from .base import BaseAgent


class GreedyAgent(BaseAgent):
    """
    Greedy AI agent that uses heuristics to evaluate board positions.
    
    This agent looks one move ahead and selects the move that results in
    the highest heuristic score. It combines multiple evaluation criteria:
    - Monotonicity: Prefers ordered tile arrangements
    - Smoothness: Minimizes differences between adjacent tiles  
    - Free tiles: Values empty spaces
    - Max tile positioning: Keeps highest tile in corners
    - Merge potential: Counts available merge opportunities
    """
    
    def __init__(self, name: str = "GreedyAgent"):
        """
        Initialize the greedy agent with heuristic weights.
        
        Args:
            name: Display name for this agent
        """
        super().__init__(name)
        
        # Heuristic weights (tuned for good performance)
        self.weights = {
            'score': 1.0,
            'monotonicity': 0.2,
            'smoothness': 0.1, 
            'free_tiles': 0.5,
            'max_tile_corner': 0.1,
            'merge_potential': 0.3
        }
    
    def get_move(self, game_state) -> str:
        """
        Select the move that maximizes the heuristic evaluation.
        
        Args:
            game_state: Current Game2048 state
            
        Returns:
            str: Best move direction based on heuristic evaluation
            
        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = self.get_valid_moves(game_state)
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        best_move = None
        best_score = float('-inf')
        
        for move in valid_moves:
            # Simulate the move
            test_game = self.copy_game_state(game_state)
            moved = test_game.move(move)
            
            # Only consider moves that actually change the board
            if not moved:
                continue
                
            # Evaluate the resulting position
            score = self.evaluate_position(test_game.grid) + test_game.score
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def evaluate_position(self, grid: np.ndarray) -> float:
        """
        Evaluate a board position using multiple heuristics.
        
        Args:
            grid: 4x4 numpy array representing the game grid
            
        Returns:
            float: Weighted combination of heuristic scores
        """
        score = 0.0
        
        score += self.weights['monotonicity'] * self._monotonicity(grid)
        score += self.weights['smoothness'] * self._smoothness(grid)
        score += self.weights['free_tiles'] * self._free_tiles(grid)
        score += self.weights['max_tile_corner'] * self._max_tile_corner(grid)
        score += self.weights['merge_potential'] * self._merge_potential(grid)
        
        return score
    
    def _monotonicity(self, grid: np.ndarray) -> float:
        """
        Measure how monotonic (ordered) the grid is.
        
        Rewards grids where larger tiles are positioned towards edges
        and there's a general ordering pattern.
        """
        def monotonic_line(line):
            """Check monotonicity of a single line using log values."""
            # Convert to log values for easier comparison, skip zeros
            log_line = []
            for val in line:
                if val > 0:
                    log_line.append(np.log2(val))
            
            if len(log_line) <= 1:
                return 0
            
            # Calculate penalty for non-monotonic arrangements
            increasing_penalty = sum(max(0, log_line[i-1] - log_line[i]) for i in range(1, len(log_line)))
            decreasing_penalty = sum(max(0, log_line[i] - log_line[i-1]) for i in range(1, len(log_line)))
            
            # Return negative of minimum penalty (so monotonic = higher score)
            return -min(increasing_penalty, decreasing_penalty)
        
        horizontal = sum(monotonic_line(grid[i]) for i in range(4))
        vertical = sum(monotonic_line(grid[:, j]) for j in range(4))
        
        return horizontal + vertical
    
    def _smoothness(self, grid: np.ndarray) -> float:
        """
        Measure smoothness - how similar adjacent tiles are.
        
        Lower differences between adjacent tiles are better.
        Returns negative value since we want to minimize differences.
        """
        smoothness = 0.0
        
        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    current = np.log2(grid[i, j])
                    
                    # Check right neighbor
                    if j < 3 and grid[i, j+1] != 0:
                        neighbor = np.log2(grid[i, j+1])
                        smoothness -= abs(current - neighbor)
                    
                    # Check down neighbor  
                    if i < 3 and grid[i+1, j] != 0:
                        neighbor = np.log2(grid[i+1, j])
                        smoothness -= abs(current - neighbor)
        
        return smoothness
    
    def _free_tiles(self, grid: np.ndarray) -> float:
        """
        Count the number of free (empty) tiles.
        
        More empty tiles provide more flexibility for future moves.
        """
        return np.sum(grid == 0)
    
    def _max_tile_corner(self, grid: np.ndarray) -> float:
        """
        Bonus for keeping the maximum tile in a corner.
        
        Corner positions are strategic as they have fewer neighbors
        and are less likely to be moved accidentally.
        """
        max_tile = np.max(grid)
        if max_tile == 0:
            return 0
            
        corners = [grid[0, 0], grid[0, 3], grid[3, 0], grid[3, 3]]
        
        if max_tile in corners:
            return np.log2(max_tile)
        else:
            # Check if max tile is on an edge (less penalty than center)
            edges = [grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 3], 
                    grid[2, 0], grid[2, 3], grid[3, 1], grid[3, 2]]
            if max_tile in edges:
                return np.log2(max_tile) * 0.5
            else:
                return 0
    
    def _merge_potential(self, grid: np.ndarray) -> float:
        """
        Count potential merge opportunities.
        
        More possible merges indicate better positioning for
        combining tiles in future moves.
        """
        merges = 0
        
        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    # Check right neighbor
                    if j < 3 and grid[i, j] == grid[i, j+1]:
                        merges += 1
                    
                    # Check down neighbor
                    if i < 3 and grid[i, j] == grid[i+1, j]:
                        merges += 1
        
        return merges
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Update heuristic weights for tuning performance.
        
        Args:
            weights: Dictionary of heuristic names to weight values
        """
        for key, value in weights.items():
            if key in self.weights:
                self.weights[key] = value
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current heuristic weights.
        
        Returns:
            Dictionary of current weight values
        """
        return self.weights.copy()