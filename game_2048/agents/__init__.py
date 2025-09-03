"""
AI Agents package for 2048 game.

This package contains various AI implementations for playing 2048,
from simple random agents to sophisticated deep learning models.
"""

from .base import BaseAgent
from .random import RandomAgent
from .greedy import GreedyAgent
from .runner import AIRunner, GameResult

__version__ = "1.0.0"
__all__ = ["BaseAgent", "RandomAgent", "GreedyAgent", "AIRunner", "GameResult"]