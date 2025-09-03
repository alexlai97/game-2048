#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
import copy


class BaseAgent(ABC):
    """
    Abstract base class for all 2048 AI agents.
    
    This class defines the standard interface that all AI agents must implement
    to interact with the 2048 game. Agents should focus on the get_move method
    which takes the current game state and returns a valid move direction.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the agent with an optional name.
        
        Args:
            name: Optional custom name for the agent. If None, uses class name.
        """
        self.name = name or self.__class__.__name__
        self.games_played = 0
        self.total_score = 0
        self.wins = 0
        self.max_score = 0
        self.max_tile = 0
    
    @abstractmethod
    def get_move(self, game_state) -> str:
        """
        Given the current game state, return the best move to make.
        
        Args:
            game_state: Game2048 instance representing current state
            
        Returns:
            str: One of 'up', 'down', 'left', 'right'
            
        Raises:
            ValueError: If no valid moves are available
        """
        pass
    
    def get_name(self) -> str:
        """Return the agent's display name."""
        return self.name
    
    def reset(self) -> None:
        """
        Reset agent's internal state.
        
        Called before starting a new game. Agents can override this
        to reset any learning state, statistics, or cached values.
        """
        pass
    
    def game_finished(self, final_score: int, won: bool, max_tile: int) -> None:
        """
        Called when a game finishes to update statistics.
        
        Args:
            final_score: The final score achieved
            won: Whether the game was won (reached 2048)
            max_tile: Highest tile value achieved
        """
        self.games_played += 1
        self.total_score += final_score
        if won:
            self.wins += 1
        if final_score > self.max_score:
            self.max_score = final_score
        if max_tile > self.max_tile:
            self.max_tile = max_tile
    
    def get_statistics(self) -> dict:
        """
        Get performance statistics for this agent.
        
        Returns:
            dict: Statistics including win rate, average score, etc.
        """
        if self.games_played == 0:
            return {
                'games_played': 0,
                'win_rate': 0.0,
                'average_score': 0.0,
                'max_score': 0,
                'max_tile': 0,
                'total_score': 0
            }
        
        return {
            'games_played': int(self.games_played),
            'win_rate': float(self.wins / self.games_played),
            'average_score': float(self.total_score / self.games_played),
            'max_score': int(self.max_score),
            'max_tile': int(self.max_tile),
            'total_score': int(self.total_score)
        }
    
    def copy_game_state(self, game_state):
        """
        Create a deep copy of the game state for simulation.
        
        Args:
            game_state: Game2048 instance to copy
            
        Returns:
            Game2048: New game instance with copied state
        """
        # Import here to avoid circular imports
        from ..core import Game2048
        
        new_game = Game2048()
        new_game.grid = game_state.grid.copy()
        new_game.score = game_state.score
        new_game.moved = game_state.moved
        return new_game
    
    def get_valid_moves(self, game_state) -> List[str]:
        """
        Get list of valid moves for the current game state.
        
        Args:
            game_state: Game2048 instance
            
        Returns:
            List[str]: List of valid move directions
        """
        return game_state.get_available_moves()
    
    def evaluate_position(self, grid: np.ndarray) -> float:
        """
        Optional method to evaluate a board position.
        
        Base implementation returns 0. Agents can override this
        to implement position evaluation heuristics.
        
        Args:
            grid: 4x4 numpy array representing the game grid
            
        Returns:
            float: Position evaluation score (higher = better)
        """
        return 0.0
    
    def __str__(self) -> str:
        """String representation of the agent."""
        stats = self.get_statistics()
        return (f"{self.name}: {stats['games_played']} games, "
                f"{stats['win_rate']:.1%} win rate, "
                f"avg score: {stats['average_score']:.1f}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"