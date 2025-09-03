# AI Agents for 2048

This document provides an overview of the AI agents implemented for the 2048 game.

## Implemented Agents

### RandomAgent
- **Strategy**: Selects moves randomly from available options
- **Purpose**: Baseline for performance comparison
- **Performance**: Average score ~1,128, reaches 128 tiles in 70% of games

### GreedyAgent
- **Strategy**: Uses heuristics to evaluate board positions and selects the best immediate move
- **Heuristics**:
  - Score maximization (immediate rewards)
  - Monotonicity (ordered tile arrangements)
  - Smoothness (adjacent tile similarity)
  - Free tiles (preserving empty spaces)
  - Max tile positioning (keeping largest tiles in corners/edges)
  - Merge potential (available combinations)
- **Performance**: Average score ~5,638, reaches 512 tiles in 70% of games
- **Improvement**: **5x better** than RandomAgent baseline

## Performance Comparison

| Agent | Avg Score | Max Tile Distribution | Win Rate |
|-------|-----------|----------------------|----------|
| Random | 1,128 | 128: 70%, 64: 30% | 0% |
| Greedy | 5,638 | 512: 70%, 256: 20%, 1024: 5% | 0% |

## Usage

```bash
# Test RandomAgent with 50 games
2048-ai RandomAgent 50

# Test GreedyAgent with 50 games
2048-ai GreedyAgent 50
```

Results are automatically saved to JSON files in the `results/` directory with performance statistics and individual game data.

## Future Agents

Planned implementations include:
- Minimax with Alpha-Beta pruning
- Monte Carlo Tree Search (MCTS)
- Expectimax algorithm
- Deep Q-Learning (DQN)
- Advanced Deep RL approaches
