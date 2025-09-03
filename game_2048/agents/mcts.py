#!/usr/bin/env python3

import math
import random
import time
from typing import Optional

import numpy as np

from .base import BaseAgent


class MCTSNode:
    """
    Node in the MCTS tree.

    Represents a game state with statistics for selection and backpropagation.
    """

    def __init__(self, game_state, parent=None, move_from_parent: Optional[str] = None):
        """
        Initialize an MCTS node.

        Args:
            game_state: The Game2048 state this node represents
            parent: Parent node (None for root)
            move_from_parent: Move that led from parent to this node
        """
        self.game_state = game_state
        self.parent = parent
        self.move_from_parent = move_from_parent

        # MCTS statistics
        self.visits = 0
        self.total_score = 0.0
        self.children: dict[str, MCTSNode] = {}

        # Track unexplored moves
        if hasattr(game_state, "get_valid_moves"):
            # Use agent's get_valid_moves method
            from .base import BaseAgent

            agent = BaseAgent()
            self.untried_moves = agent.get_valid_moves(game_state)
        else:
            # Fallback to manual check
            self.untried_moves = []
            for move in ["up", "down", "left", "right"]:
                test_game = self._copy_game_state(game_state)
                if test_game.move(move):
                    self.untried_moves.append(move)

    def _copy_game_state(self, game_state):
        """Create a deep copy of game state."""
        from game_2048 import Game2048

        new_game = Game2048()
        new_game.grid = game_state.grid.copy()
        new_game.score = game_state.score
        return new_game

    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)."""
        return self.game_state.is_game_over()

    def ucb1_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """
        Calculate UCB1 value for node selection.

        Args:
            exploration_constant: Exploration vs exploitation balance (usually √2)

        Returns:
            float: UCB1 value for this node
        """
        if self.visits == 0:
            return float("inf")

        exploitation = self.total_score / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def select_best_child(
        self, exploration_constant: float = math.sqrt(2)
    ) -> "MCTSNode":
        """
        Select child with highest UCB1 value.

        Args:
            exploration_constant: Exploration parameter

        Returns:
            MCTSNode: Child node with highest UCB1 value
        """
        return max(
            self.children.values(),
            key=lambda child: child.ucb1_value(exploration_constant),
        )

    def expand(self) -> "MCTSNode":
        """
        Expand by adding a new child for an untried move.

        Returns:
            MCTSNode: The newly created child node
        """
        if not self.untried_moves:
            raise ValueError("Cannot expand: no untried moves")

        move = self.untried_moves.pop(0)

        # Create new game state by applying the move
        child_game = self._copy_game_state(self.game_state)
        move_success = child_game.move(move)

        if not move_success:
            # Move failed, try next move
            if self.untried_moves:
                return self.expand()
            else:
                raise ValueError("No valid moves to expand")

        # Create child node
        child_node = MCTSNode(child_game, parent=self, move_from_parent=move)
        self.children[move] = child_node

        return child_node

    def update(self, score: float) -> None:
        """
        Update node statistics with simulation result.

        Args:
            score: Score from the simulation
        """
        self.visits += 1
        self.total_score += score

    def best_move(self) -> str:
        """Get the move to the child with most visits (most promising)."""
        if not self.children:
            raise ValueError("No children to select from")

        return max(self.children.items(), key=lambda item: item[1].visits)[0]


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search AI agent for 2048.

    MCTS builds a search tree by iteratively:
    1. Selection: Navigate to promising leaf using UCB1
    2. Expansion: Add new child node
    3. Simulation: Random rollout to estimate value
    4. Backpropagation: Update statistics up the tree

    This approach balances exploration of new moves with exploitation
    of promising paths through the UCB1 selection policy.
    """

    def __init__(
        self,
        name: str = "MCTSAgent",
        simulations: int = 25,
        exploration_constant: float = math.sqrt(2),
    ):
        """
        Initialize the MCTS agent.

        Args:
            name: Display name for this agent
            simulations: Number of MCTS simulations per move
            exploration_constant: UCB1 exploration parameter (usually √2)
        """
        super().__init__(name)

        self.simulations = simulations
        self.exploration_constant = exploration_constant

        # Performance tracking
        self.total_simulations = 0
        self.total_time = 0.0

    def get_move(self, game_state) -> str:
        """
        Select the best move using Monte Carlo Tree Search.

        Args:
            game_state: Current Game2048 state

        Returns:
            str: Best move direction based on MCTS analysis

        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            raise ValueError("No valid moves available")

        if len(valid_moves) == 1:
            return valid_moves[0]

        start_time = time.time()

        # Create root node
        root = MCTSNode(game_state)

        # Run MCTS simulations
        for _ in range(self.simulations):
            self._mcts_iteration(root)

        # Track performance
        self.total_simulations += self.simulations
        self.total_time += time.time() - start_time

        # Return move with most visits
        return root.best_move()

    def _mcts_iteration(self, root: MCTSNode) -> None:
        """
        Perform one MCTS iteration: Selection, Expansion, Simulation, Backpropagation.

        Args:
            root: Root node of the search tree
        """
        # 1. Selection - navigate to leaf using UCB1
        node = self._select(root)

        # 2. Expansion - add child if not terminal
        if not node.is_terminal() and node.visits > 0:
            if not node.is_fully_expanded():
                node = node.expand()

        # 3. Simulation - random rollout from current node
        score = self._simulate(node.game_state)

        # 4. Backpropagation - update statistics up the tree
        self._backpropagate(node, score)

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Selection phase: navigate down tree using UCB1 until leaf.

        Args:
            root: Root node to start selection from

        Returns:
            MCTSNode: Selected leaf node
        """
        node = root

        while not node.is_terminal() and node.is_fully_expanded() and node.children:
            node = node.select_best_child(self.exploration_constant)

        return node

    def _simulate(self, game_state) -> float:
        """
        Simulation phase: random rollout to estimate position value.

        Args:
            game_state: Starting game state for simulation

        Returns:
            float: Estimated value of the position
        """
        # Create copy for simulation
        sim_game = self._copy_game_state(game_state)

        # Random rollout with depth limit
        max_depth = 20
        depth = 0

        while not sim_game.is_game_over() and depth < max_depth:
            valid_moves = self.get_valid_moves(sim_game)
            if not valid_moves:
                break

            # Choose random move with slight bias toward good heuristics
            if len(valid_moves) > 1:
                # Quick heuristic scoring for move selection
                move_scores = []
                for move in valid_moves:
                    test_game = self._copy_game_state(sim_game)
                    if test_game.move(move):
                        # Simple heuristic: score gain + empty tiles
                        score_gain = test_game.score - sim_game.score
                        empty_tiles = np.sum(test_game.grid == 0)
                        heuristic_score = score_gain + empty_tiles * 10
                        move_scores.append((move, heuristic_score))
                    else:
                        move_scores.append((move, -1000))

                # Weighted random selection (80% best move, 20% random)
                if random.random() < 0.8 and move_scores:
                    # Select best move
                    best_move = max(move_scores, key=lambda x: x[1])[0]
                    move = best_move
                else:
                    # Random move
                    move = random.choice(valid_moves)
            else:
                move = valid_moves[0]

            sim_game.move(move)
            depth += 1

        # Return evaluation based on final state
        return self._evaluate_final_state(sim_game)

    def _evaluate_final_state(self, game_state) -> float:
        """
        Evaluate the final state of a simulation.

        Args:
            game_state: Final game state

        Returns:
            float: Evaluation score
        """
        score = game_state.score

        # Bonus for higher tiles
        max_tile = np.max(game_state.grid)
        if max_tile > 0:
            score += np.log2(max_tile) * 100

        # Bonus for empty tiles (maintaining flexibility)
        empty_tiles = np.sum(game_state.grid == 0)
        score += empty_tiles * 50

        # Win bonus
        if max_tile >= 2048:
            score += 10000

        return score

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """
        Backpropagation phase: update statistics up the tree.

        Args:
            node: Starting node for backpropagation
            score: Score to propagate up the tree
        """
        current = node

        while current is not None:
            current.update(score)
            current = current.parent

    def _copy_game_state(self, game_state):
        """Create a deep copy of game state."""
        from game_2048 import Game2048

        new_game = Game2048()
        new_game.grid = game_state.grid.copy()
        new_game.score = game_state.score
        return new_game

    def set_simulations(self, simulations: int) -> None:
        """Set the number of simulations per move."""
        self.simulations = max(1, simulations)

    def set_exploration_constant(self, constant: float) -> None:
        """Set the UCB1 exploration constant."""
        self.exploration_constant = max(0.1, constant)

    def get_performance_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        if self.total_simulations == 0:
            return {"avg_time_per_simulation": 0.0, "total_simulations": 0}

        return {
            "avg_time_per_simulation": self.total_time / self.total_simulations,
            "total_simulations": self.total_simulations,
            "total_time": self.total_time,
        }
