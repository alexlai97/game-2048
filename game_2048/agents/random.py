#!/usr/bin/env python3

import random
from .base import BaseAgent


class RandomAgent(BaseAgent):
    """
    Random AI agent that selects moves randomly from available options.
    
    This agent serves as a baseline for performance comparison with other
    AI strategies. It makes no attempt to play strategically and simply
    chooses valid moves at random.
    """
    
    def __init__(self, name: str = "RandomAgent"):
        """
        Initialize the random agent.
        
        Args:
            name: Display name for this agent
        """
        super().__init__(name)
        self.random_seed = None
    
    def get_move(self, game_state) -> str:
        """
        Select a random valid move from available options.
        
        Args:
            game_state: Current Game2048 state
            
        Returns:
            str: Randomly selected valid move direction
            
        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = self.get_valid_moves(game_state)
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        return random.choice(valid_moves)
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible results.
        
        Args:
            seed: Random seed value
        """
        self.random_seed = seed
        random.seed(seed)
    
    def reset(self) -> None:
        """Reset agent state (re-apply random seed if set)."""
        super().reset()
        if self.random_seed is not None:
            random.seed(self.random_seed)