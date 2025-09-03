# Project Architecture

This document provides an overview of the 2048 AI learning project structure.

## Project Structure

```
game-2048/
├── game_2048/              # Main game package
│   ├── core.py            # Game logic and mechanics
│   ├── terminal.py        # Terminal-based UI
│   ├── gui.py             # PySide6 GUI with AI visualization
│   ├── agents/            # AI agent implementations
│   │   ├── base.py        # BaseAgent abstract class
│   │   ├── random.py      # RandomAgent (baseline)
│   │   ├── greedy.py      # GreedyAgent (heuristic-based)
│   │   ├── minimax.py     # MinimaxAgent (tree search)
│   │   ├── expectimax.py  # ExpectimaxAgent (probabilistic)
│   │   ├── mcts.py        # MCTSAgent (Monte Carlo Tree Search)
│   │   └── runner.py      # AI testing and batch runner
│   └── sounds/            # Audio effects for GUI
├── tests/                 # Unit tests for game logic
├── docs/                  # Architecture documentation
├── CLAUDE.md             # Complete project documentation
├── TODOs.md              # Project roadmap and tasks
├── requirements.txt       # Runtime dependencies
└── requirements-dev.txt   # Development dependencies
```

## Core Components

### Game Logic (`core.py`)
- **Game2048 class**: Implements 2048 game mechanics
- **Grid operations**: 4x4 NumPy array for efficient tile manipulation
- **Move mechanics**: Sliding, merging, and tile spawning
- **State detection**: Game over and win conditions

### AI Agent Framework (`agents/`)
- **BaseAgent**: Abstract base class defining the AI interface
- **Five implemented agents**: Random, Greedy, Minimax, Expectimax, MCTS
- **Statistics tracking**: Performance metrics and game results
- **Agent interface**: `get_move(game_state) -> str` method
- **Utility methods**: Game state copying, move validation, batch testing

### User Interfaces
- **Terminal UI**: Text-based interface for human play (`terminal.py`)
- **GUI**: PySide6-based graphical interface with AI visualization (`gui.py`)
  - Real-time AI agent play with speed controls
  - Agent selection dropdown for all 5 AI agents
  - Move highlighting and sound effects
  - Performance statistics display
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

# AI testing - Available agents: RandomAgent, GreedyAgent, MinimaxAgent, ExpectimaxAgent, MCTSAgent
2048-ai <AgentName> <NumGames>
```

## Dependencies

### Runtime Dependencies
- **NumPy**: Efficient grid operations and matrix calculations
- **PySide6**: Modern GUI framework with AI visualization
- **Standard library**: Core game logic uses only Python standard library

### Development Dependencies
- **Ruff**: Ultra-fast Python linter and formatter
- **Mypy**: Static type checking
- **Black**: Code formatting
- **Pre-commit**: Automated code quality checks

## Design Patterns

- **Strategy Pattern**: AI agents implement a common interface (`BaseAgent`)
- **Observer Pattern**: GUI updates based on game state changes
- **Factory Pattern**: Agent creation based on string names
- **Template Method**: Base agent provides common utilities, subclasses implement specific strategies
