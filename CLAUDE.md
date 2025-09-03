# 2048 AI Learning Project

## Project Overview
Building a 2048 game implementation followed by various AI agents to solve it. This is a learning project to explore different machine learning strategies.

## Project Structure
- `game_2048/core.py` - Core game logic and mechanics
- `game_2048/terminal.py` - Terminal UI for human play
- `game_2048/gui.py` - PySide6 GUI with AI visualization
- `game_2048/agents/` - AI agent implementations
  - `base.py` - Base agent class and framework
  - `random.py` - Random baseline agent
  - `greedy.py` - Heuristic-based greedy agent
  - `minimax.py` - Minimax with alpha-beta pruning
  - `expectimax.py` - Expectimax algorithm
  - `mcts.py` - Monte Carlo Tree Search
  - `runner.py` - AI testing and evaluation framework
- `TODOs.md` - Task tracking
- `requirements.txt` - Python dependencies

## Game Implementation Notes
- Use NumPy for efficient 4x4 grid operations
- Game state: 4x4 matrix with powers of 2
- Move mechanics: slide all tiles in direction, merge adjacent identical tiles
- Score: sum of merged tile values
- Win condition: reach 2048 tile
- Lose condition: no valid moves available

## AI Strategy Learning Path
1. **Random Agent** - Baseline performance
2. **Greedy Search** - Simple heuristics
3. **Tree Search** - Minimax, MCTS, Expectimax
4. **Deep Learning** - DQN, Policy Gradient methods

## Development Setup
```bash
# Create virtual environment
uv venv

# Install runtime dependencies
source .venv/bin/activate && uv pip install -r requirements.txt

# Install development dependencies (linting, formatting, type checking)
source .venv/bin/activate && uv pip install -r requirements-dev.txt

# Install pre-commit hooks for automatic code quality checks
source .venv/bin/activate && pre-commit install
```

## Testing Commands
```bash
# Install in development mode
source .venv/bin/activate
uv pip install -e .

# Terminal version
source .venv/bin/activate
2048-terminal
# or: python -m game_2048.terminal

# PySide6 GUI
source .venv/bin/activate
2048-gui
# or: python -m game_2048.gui

# Test game logic
source .venv/bin/activate
python -c "from game_2048 import Game2048; g = Game2048(); print(g)"

# AI testing - Available agents: RandomAgent, GreedyAgent, MinimaxAgent, ExpectimaxAgent, MCTSAgent
source .venv/bin/activate
2048-ai RandomAgent 100           # Random baseline
2048-ai GreedyAgent 100           # Heuristic greedy search
2048-ai MinimaxAgent 10           # Minimax with alpha-beta pruning (slower)
2048-ai ExpectimaxAgent 10        # Expectimax algorithm (slower)
2048-ai MCTSAgent 10              # Monte Carlo Tree Search (slower)
# or: python -m game_2048.agents.runner <AgentName> <num_games>

# Run tests
source .venv/bin/activate
pytest tests/

# Code Quality Commands
source .venv/bin/activate
ruff check game_2048/ --fix          # Lint and auto-fix code issues
ruff format game_2048/               # Format code with consistent style
mypy game_2048/                      # Type checking
pre-commit run --all-files           # Run all quality checks
```

## Implementation Status
✅ **Phase 1: Game Implementation (Complete)**
- Core game logic with NumPy 4x4 grid
- Tile sliding, merging, and spawning mechanics
- Score tracking and game state detection
- Terminal-based user interface
- Move validation and available moves detection

✅ **Phase 2: AI Agents (Complete - Tree Search Algorithms)**
- Random baseline agent ✅
- AI framework and runner ✅
- Greedy heuristic-based agent ✅
- Minimax with alpha-beta pruning ✅
- Expectimax algorithm ✅
- Monte Carlo Tree Search (MCTS) ✅
- GUI integration with all agents ✅

✅ **Phase 3: Code Quality Infrastructure (Complete)**
- Ruff linting and formatting ✅
- Mypy type checking ✅
- Pre-commit hooks ✅
- Black code formatting ✅
- Development dependencies ✅

## Dependencies

### Runtime Dependencies
- numpy>=1.20.0: efficient array operations
- PySide6>=6.6.0: GUI framework
- pytest>=6.0.0: testing framework

### Development Dependencies (Code Quality)
- ruff>=0.12.0: ultra-fast Python linter and formatter
- black>=25.0.0: opinionated code formatter
- mypy>=1.17.0: static type checker
- pre-commit>=4.0.0: git hooks for automated quality checks

### Future Dependencies
- (future) torch/tensorflow: deep learning
- (future) matplotlib: visualization

## Code Quality Setup (2025 Best Practices)

This project uses modern Python code quality tools:

- **Ruff**: Replaces Flake8, isort, pyupgrade, and more - 10-100x faster
- **Black**: Consistent code formatting (88-character line length)
- **Mypy**: Static type checking with strict configuration
- **Pre-commit hooks**: Automatic quality checks on every commit

All tools are configured in `pyproject.toml` with compatible settings. Pre-commit hooks automatically run linting, formatting, and basic file quality checks before each commit.

## AI Agents Overview

### Implemented Agents

**1. RandomAgent** - Baseline Performance
- Chooses random valid moves
- Average score: ~1,000-2,000
- Serves as performance baseline
- Fast execution: <1ms per move

**2. GreedyAgent** - Heuristic Search
- Uses multiple weighted heuristics:
  - Monotonicity (ordered tile arrangements)
  - Smoothness (similar adjacent tiles)
  - Free tiles (empty space availability)
  - Max tile positioning
- Average score: ~8,000-12,000
- Reaches 1024 tile consistently
- Fast execution: <1ms per move

**3. MinimaxAgent** - Game Tree Search
- Minimax with alpha-beta pruning
- Models player moves (MAX) vs random tile spawns (CHANCE)
- Configurable search depth (default: 3)
- Average score: ~4,000-6,000
- Medium execution: ~10-30s per move

**4. ExpectimaxAgent** - Probabilistic Tree Search
- Expectimax algorithm with chance node modeling
- Better suited for stochastic games than Minimax
- Probabilistic evaluation of random tile spawns
- Average score: ~6,000-8,000
- Medium execution: ~15-40s per move

**5. MCTSAgent** - Monte Carlo Tree Search
- UCB1 selection policy for exploration/exploitation
- Random rollouts with heuristic guidance
- Configurable simulation count (default: 25)
- Average score: ~4,000-6,000
- Slow execution: ~20-60s per move

### Performance Comparison
| Agent | Avg Score | Max Tile | Speed | Best For |
|-------|-----------|----------|-------|----------|
| Random | ~1,500 | 128-256 | Very Fast | Baseline |
| Greedy | ~9,000 | 1024 | Very Fast | Production |
| Minimax | ~5,000 | 512 | Medium | Learning |
| Expectimax | ~7,000 | 512-1024 | Medium | Stochastic Games |
| MCTS | ~4,500 | 512 | Slow | Research |

### Usage Recommendations
- **For best performance**: Use GreedyAgent
- **For learning tree search**: Use MinimaxAgent or ExpectimaxAgent
- **For research/experimentation**: Use MCTSAgent
- **For GUI visualization**: Any agent (MCTS uses reduced simulations)

### Future Implementations (Phase 3)
- Deep Q-Learning (DQN)
- Policy Gradient methods
- Actor-Critic algorithms
- Advanced neural network architectures
