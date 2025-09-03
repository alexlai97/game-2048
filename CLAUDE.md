# 2048 AI Learning Project

## Project Overview
Building a 2048 game implementation followed by various AI agents to solve it. This is a learning project to explore different machine learning strategies.

## Project Structure
- `game_2048.py` - Core game logic and mechanics
- `main.py` - Terminal UI for human play
- `ai_agents/` - Directory for AI implementations (future)
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

# AI testing
source .venv/bin/activate
2048-ai RandomAgent 100
# or: python -m game_2048.agents.runner RandomAgent 100

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

✅ **Phase 2: AI Agents (In Progress)**
- Random baseline agent ✅
- AI framework and runner ✅
- Heuristic-based strategies
- Tree search algorithms
- Deep learning approaches

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
