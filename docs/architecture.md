# Project Architecture

This document provides an overview of the 2048 AI learning project structure.

## Project Structure

```
game-2048/
├── game_2048/              # Main game package
│   ├── core.py            # Game logic and mechanics
│   ├── terminal.py        # Terminal-based UI
│   ├── gui.py             # PySide6 GUI interface
│   └── agents/            # AI agent implementations
│       ├── base.py        # BaseAgent abstract class
│       ├── random.py      # RandomAgent implementation
│       ├── greedy.py      # GreedyAgent with heuristics
│       └── runner.py      # AI testing and batch runner
├── tests/                 # Unit tests for game logic
├── results/               # AI performance test results (JSON)
├── docs/                  # Documentation
├── TODOs.md              # Project roadmap and tasks
└── requirements.txt       # Python dependencies
```

## Core Components

### Game Logic (`core.py`)
- **Game2048 class**: Implements 2048 game mechanics
- **Grid operations**: 4x4 NumPy array for efficient tile manipulation
- **Move mechanics**: Sliding, merging, and tile spawning
- **State detection**: Game over and win conditions

### AI Agent Framework (`agents/`)
- **BaseAgent**: Abstract base class defining the AI interface
- **Statistics tracking**: Performance metrics and game results
- **Agent interface**: `get_move(game_state) -> str` method
- **Utility methods**: Game state copying, move validation

### User Interfaces
- **Terminal UI**: Text-based interface for human play (`terminal.py`)
- **GUI**: PySide6-based graphical interface (`gui.py`)
- **AI Runner**: Command-line tool for batch testing agents (`runner.py`)

## How AI Agents Work

1. **Agent Interface**: All agents inherit from `BaseAgent` and implement `get_move()`
2. **Game Simulation**: Agents can copy game state to test moves without affecting the real game
3. **Move Selection**: Agents analyze the current board and return one of: 'up', 'down', 'left', 'right'
4. **Performance Tracking**: Built-in statistics collection for win rate, average score, etc.

## Entry Points

```bash
# Human play (terminal)
2048-terminal

# Human play (GUI)
2048-gui

# AI testing
2048-ai <AgentName> <NumGames>
```

## Dependencies

- **NumPy**: Efficient grid operations
- **PySide6**: GUI framework
- **Standard library**: No additional external dependencies for core logic
