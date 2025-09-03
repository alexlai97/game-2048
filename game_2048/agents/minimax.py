#!/usr/bin/env python3

import random

import numpy as np

from .base import BaseAgent


class MinimaxAgent(BaseAgent):
    """
    Minimax AI agent with Alpha-Beta pruning for 2048.

    This agent uses a minimax search algorithm to look ahead multiple moves,
    modeling both player moves (MAX nodes) and random tile spawns (CHANCE nodes).
    Alpha-Beta pruning is used to reduce the search space and improve performance.

    The evaluation function combines multiple heuristics:
    - Monotonicity: Ordered tile arrangements
    - Smoothness: Similar adjacent tiles
    - Free tiles: Available empty spaces
    - Max tile positioning: Keep largest tile in corner
    - Merge potential: Available merge opportunities
    - Weighted positions: Strategic tile placement
    """

    def __init__(
        self, name: str = "MinimaxAgent", depth: int = 3, sampling_ratio: float = 0.8
    ):
        """
        Initialize the Minimax agent.

        Args:
            name: Display name for this agent
            depth: Maximum search depth (higher = better play but slower)
            sampling_ratio: Fraction of spawn positions to evaluate (0.5-1.0)
        """
        super().__init__(name)

        self.max_depth = depth
        self.sampling_ratio = sampling_ratio
        self.nodes_evaluated = 0  # For performance monitoring

        # Optimized heuristic weights (tuned for minimax with reduced depth)
        self.weights = {
            "monotonicity": 5.0,  # More important for strategic positioning
            "smoothness": 0.5,  # Slightly more weight for tile adjacency
            "free_tiles": 4.0,  # Critical for maintaining flexibility
            "max_tile_corner": 2.0,  # Important for endgame strategy
            "merge_potential": 2.5,  # Higher weight for available merges
            "weighted_positions": 3.0,  # Strategic tile placement
        }

        # Position weights favor corners and edges
        self.position_weights = np.array(
            [
                [3.0, 2.0, 2.0, 3.0],
                [2.0, 1.0, 1.0, 2.0],
                [2.0, 1.0, 1.0, 2.0],
                [3.0, 2.0, 2.0, 3.0],
            ]
        )

    def get_move(self, game_state) -> str:
        """
        Select the best move using minimax search with alpha-beta pruning.

        Args:
            game_state: Current Game2048 state

        Returns:
            str: Best move direction based on minimax search

        Raises:
            ValueError: If no valid moves are available
        """
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            raise ValueError("No valid moves available")

        best_move = None
        best_score = float("-inf")

        # Reset performance counter
        self.nodes_evaluated = 0

        # Order moves by quick heuristic for better search
        moves_with_quick_scores = []
        for move in valid_moves:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                quick_score = (
                    test_game.score
                    - game_state.score  # Score gained
                    + self._free_tiles(test_game.grid) * 50  # Value empty spaces
                    + self._merge_potential(test_game.grid) * 100
                )  # Value merge opportunities
                moves_with_quick_scores.append((move, quick_score))

        # Sort moves by quick score (best first) for better pruning
        moves_with_quick_scores.sort(key=lambda x: x[1], reverse=True)

        # Try each move in order
        for move, _ in moves_with_quick_scores:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                # Evaluate this move using minimax (this will be a CHANCE node)
                score = self._minimax_chance_node(
                    test_game, self.max_depth - 1, float("-inf"), float("inf")
                )

                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    def _minimax_max_node(
        self, game_state, depth: int, alpha: float, beta: float
    ) -> float:
        """
        Evaluate a MAX node (player's turn to move).

        Args:
            game_state: Current game state
            depth: Remaining search depth
            alpha: Current alpha value for pruning
            beta: Current beta value for pruning

        Returns:
            float: Best score achievable from this position
        """
        # Terminal conditions
        if depth == 0 or game_state.is_game_over():
            self.nodes_evaluated += 1
            return self.evaluate_position(game_state.grid)

        max_eval = float("-inf")
        valid_moves = self.get_valid_moves(game_state)

        # Order moves by quick heuristic for better pruning
        moves_with_scores = []
        for move in valid_moves:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                quick_score = test_game.score + self._free_tiles(test_game.grid) * 10
                moves_with_scores.append((move, quick_score))

        # Sort moves by quick score (best first) for better pruning
        moves_with_scores.sort(key=lambda x: x[1], reverse=True)

        for move, _ in moves_with_scores:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                eval_score = self._minimax_chance_node(
                    test_game, depth - 1, alpha, beta
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                # Alpha-beta pruning
                if beta <= alpha:
                    break

        # Early termination if clearly winning position
        if max_eval > 50000:
            return max_eval

        return max_eval

    def _minimax_chance_node(
        self, game_state, depth: int, alpha: float, beta: float
    ) -> float:
        """
        Evaluate a CHANCE node (random tile spawn).

        Args:
            game_state: Current game state (after player move)
            depth: Remaining search depth
            alpha: Current alpha value for pruning
            beta: Current beta value for pruning

        Returns:
            float: Expected score from this position
        """
        # Terminal condition
        if depth == 0:
            self.nodes_evaluated += 1
            return self.evaluate_position(game_state.grid)

        empty_cells = [
            (i, j) for i in range(4) for j in range(4) if game_state.grid[i, j] == 0
        ]

        if not empty_cells:
            return self.evaluate_position(game_state.grid)

        # More aggressive sampling for better performance
        if len(empty_cells) > 4:
            sample_size = max(2, min(4, int(len(empty_cells) * self.sampling_ratio)))
            # Prioritize corner and edge cells for sampling
            corner_edge_cells = [
                (i, j) for i, j in empty_cells if i in [0, 3] or j in [0, 3]
            ]
            if len(corner_edge_cells) >= sample_size:
                empty_cells = random.sample(corner_edge_cells, sample_size)
            else:
                remaining_needed = sample_size - len(corner_edge_cells)
                other_cells = [
                    (i, j) for i, j in empty_cells if (i, j) not in corner_edge_cells
                ]
                if other_cells:
                    empty_cells = corner_edge_cells + random.sample(
                        other_cells, min(remaining_needed, len(other_cells))
                    )
                else:
                    empty_cells = corner_edge_cells

        expected_score = 0.0
        probability_mass = 0.0

        for i, j in empty_cells:
            # Try spawning a 2 tile (90% probability)
            test_game = self.copy_game_state(game_state)
            test_game.grid[i, j] = 2
            score_2 = self._minimax_max_node(test_game, depth - 1, alpha, beta)
            expected_score += 0.9 * score_2
            probability_mass += 0.9

            # Try spawning a 4 tile (10% probability)
            test_game.grid[i, j] = 4
            score_4 = self._minimax_max_node(test_game, depth - 1, alpha, beta)
            expected_score += 0.1 * score_4
            probability_mass += 0.1

        # Normalize by the total probability mass of sampled cells
        if probability_mass > 0:
            expected_score /= len(empty_cells)

        return expected_score

    def evaluate_position(self, grid: np.ndarray) -> float:
        """
        Evaluate a board position using multiple heuristics.

        Args:
            grid: 4x4 numpy array representing the game grid

        Returns:
            float: Weighted combination of heuristic scores
        """
        # Quick game over check
        if np.sum(grid == 0) == 0 and not self._has_adjacent_equal(grid):
            # Game over position - heavily penalize
            return -100000

        # Quick win condition bonus
        if np.max(grid) >= 2048:
            return 100000

        score = 0.0

        # Calculate all heuristics
        monotonicity = self._monotonicity(grid)
        smoothness = self._smoothness(grid)
        free_tiles = self._free_tiles(grid)
        max_tile_corner = self._max_tile_corner(grid)
        merge_potential = self._merge_potential(grid)
        weighted_positions = self._weighted_positions(grid)

        # Combine with weights
        score += self.weights["monotonicity"] * monotonicity
        score += self.weights["smoothness"] * smoothness
        score += self.weights["free_tiles"] * free_tiles
        score += self.weights["max_tile_corner"] * max_tile_corner
        score += self.weights["merge_potential"] * merge_potential
        score += self.weights["weighted_positions"] * weighted_positions

        # Bonus for high-value tiles (exponential reward)
        max_tile = np.max(grid)
        if max_tile > 0:
            score += np.log2(max_tile) * 100

        return score

    def _has_adjacent_equal(self, grid: np.ndarray) -> bool:
        """Check if any adjacent tiles are equal (merge possible)."""
        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    if (j < 3 and grid[i, j] == grid[i, j + 1]) or (
                        i < 3 and grid[i, j] == grid[i + 1, j]
                    ):
                        return True
        return False

    def _monotonicity(self, grid: np.ndarray) -> float:
        """Measure monotonicity using log values."""

        def monotonic_line(line):
            log_line = [np.log2(val) for val in line if val > 0]

            if len(log_line) <= 1:
                return 0

            increasing_penalty = sum(
                max(0, log_line[i - 1] - log_line[i]) for i in range(1, len(log_line))
            )
            decreasing_penalty = sum(
                max(0, log_line[i] - log_line[i - 1]) for i in range(1, len(log_line))
            )

            return -min(increasing_penalty, decreasing_penalty)

        horizontal = sum(monotonic_line(grid[i]) for i in range(4))
        vertical = sum(monotonic_line(grid[:, j]) for j in range(4))

        return horizontal + vertical

    def _smoothness(self, grid: np.ndarray) -> float:
        """Measure smoothness between adjacent tiles."""
        smoothness = 0.0

        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    current = np.log2(grid[i, j])

                    if j < 3 and grid[i, j + 1] != 0:
                        neighbor = np.log2(grid[i, j + 1])
                        smoothness -= abs(current - neighbor)

                    if i < 3 and grid[i + 1, j] != 0:
                        neighbor = np.log2(grid[i + 1, j])
                        smoothness -= abs(current - neighbor)

        return smoothness

    def _free_tiles(self, grid: np.ndarray) -> float:
        """Count free tiles."""
        return np.sum(grid == 0)

    def _max_tile_corner(self, grid: np.ndarray) -> float:
        """Bonus for max tile in corner."""
        max_tile = np.max(grid)
        if max_tile == 0:
            return 0

        corners = [grid[0, 0], grid[0, 3], grid[3, 0], grid[3, 3]]

        if max_tile in corners:
            return np.log2(max_tile) * 2

        # Lesser bonus for edges
        edges = [
            grid[0, 1],
            grid[0, 2],
            grid[1, 0],
            grid[1, 3],
            grid[2, 0],
            grid[2, 3],
            grid[3, 1],
            grid[3, 2],
        ]
        if max_tile in edges:
            return np.log2(max_tile)

        return 0

    def _merge_potential(self, grid: np.ndarray) -> float:
        """Count potential merges."""
        merges = 0

        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    if j < 3 and grid[i, j] == grid[i, j + 1]:
                        merges += 1
                    if i < 3 and grid[i, j] == grid[i + 1, j]:
                        merges += 1

        return merges

    def _weighted_positions(self, grid: np.ndarray) -> float:
        """Weight tiles by their strategic position."""
        score = 0.0

        for i in range(4):
            for j in range(4):
                if grid[i, j] != 0:
                    score += np.log2(grid[i, j]) * self.position_weights[i, j]

        return score

    def set_depth(self, depth: int) -> None:
        """Set the maximum search depth."""
        self.max_depth = max(1, depth)

    def set_sampling_ratio(self, ratio: float) -> None:
        """Set the chance node sampling ratio."""
        self.sampling_ratio = max(0.1, min(1.0, ratio))
