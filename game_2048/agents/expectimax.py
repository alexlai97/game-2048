#!/usr/bin/env python3


import numpy as np

from .base import BaseAgent


class ExpectimaxAgent(BaseAgent):
    """
    Expectimax AI agent for 2048.

    This agent uses the Expectimax algorithm to look ahead multiple moves,
    modeling player moves as MAX nodes and random tile spawns as CHANCE nodes.
    Unlike Minimax, Expectimax correctly models the probabilistic nature of
    tile spawning in 2048, making it theoretically more suitable for this game.

    The evaluation function combines multiple heuristics:
    - Monotonicity: Ordered tile arrangements
    - Smoothness: Similar adjacent tiles
    - Free tiles: Available empty spaces
    - Max tile positioning: Keep largest tile in corner
    - Merge potential: Available merge opportunities
    - Weighted positions: Strategic tile placement
    """

    def __init__(
        self, name: str = "ExpectimaxAgent", depth: int = 3, sampling_ratio: float = 0.7
    ):
        """
        Initialize the Expectimax agent.

        Args:
            name: Display name for this agent
            depth: Maximum search depth (higher = better play but slower)
            sampling_ratio: Fraction of empty cells to evaluate in chance nodes (0.1-1.0)
        """
        super().__init__(name)

        self.max_depth = depth
        self.sampling_ratio = sampling_ratio
        self.nodes_evaluated = 0  # For performance monitoring

        # Heuristic weights optimized for Expectimax (tuned for better performance)
        self.weights = {
            "monotonicity": 8.0,  # Much higher - strategic positioning is critical
            "smoothness": 1.2,  # Increased - tile adjacency matters more
            "free_tiles": 15.0,  # Tripled - keeping options open is crucial
            "max_tile_corner": 6.0,  # Increased - corner strategy essential
            "merge_potential": 4.5,  # Increased - available merges key to progress
            "weighted_positions": 5.0,  # Increased - strategic placement important
            "score_bonus": 2.0,  # New - reward actual score gains
            "tile_clustering": -3.0,  # New - penalize scattered high tiles
        }

        # Position weights strongly favor corners and edges (snake-like pattern)
        self.position_weights = np.array(
            [
                [15.0, 14.0, 13.0, 12.0],
                [8.0, 9.0, 10.0, 11.0],
                [7.0, 6.0, 5.0, 4.0],
                [0.0, 1.0, 2.0, 3.0],
            ]
        )

    def get_move(self, game_state) -> str:
        """
        Select the best move using Expectimax search.

        Args:
            game_state: Current Game2048 state

        Returns:
            str: Best move direction based on Expectimax search

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

        # Order moves by quick heuristic for efficiency
        moves_with_quick_scores = []
        for move in valid_moves:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                quick_score = (
                    (test_game.score - game_state.score)
                    * 2  # Score gained (higher weight)
                    + self._free_tiles(test_game.grid) * 100  # Value empty spaces more
                    + self._merge_potential(test_game.grid)
                    * 200  # Value merge opportunities highly
                    + self._monotonicity(test_game.grid)
                    * 50  # Add monotonicity to quick eval
                )
                moves_with_quick_scores.append((move, quick_score))

        # Sort moves by quick score (best first)
        moves_with_quick_scores.sort(key=lambda x: x[1], reverse=True)

        # Evaluate each move using Expectimax
        for move, _ in moves_with_quick_scores:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                # After player move, evaluate the resulting chance node
                score = self._expectimax_chance_node(test_game, self.max_depth - 1)

                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    def _expectimax_max_node(self, game_state, depth: int) -> float:
        """
        Evaluate a MAX node (player's turn to move).

        Args:
            game_state: Current game state
            depth: Remaining search depth

        Returns:
            float: Best score achievable from this position
        """
        # Terminal conditions
        if depth == 0 or game_state.is_game_over():
            self.nodes_evaluated += 1
            return self.evaluate_position(game_state.grid)

        max_eval = float("-inf")
        valid_moves = self.get_valid_moves(game_state)

        if not valid_moves:
            return self.evaluate_position(game_state.grid)

        # Order moves by quick heuristic
        moves_with_scores = []
        for move in valid_moves:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                quick_score = test_game.score + self._free_tiles(test_game.grid) * 15
                moves_with_scores.append((move, quick_score))

        # Sort moves by quick score (best first)
        moves_with_scores.sort(key=lambda x: x[1], reverse=True)

        for move, _ in moves_with_scores:
            test_game = self.copy_game_state(game_state)
            if test_game.move(move):
                eval_score = self._expectimax_chance_node(test_game, depth - 1)
                max_eval = max(max_eval, eval_score)

                # Early termination for clearly winning positions
                if max_eval > 100000:
                    break

        return max_eval

    def _expectimax_chance_node(self, game_state, depth: int) -> float:
        """
        Evaluate a CHANCE node (random tile spawn).

        Args:
            game_state: Current game state (after player move)
            depth: Remaining search depth

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

        # Intelligent sampling strategy for performance
        if len(empty_cells) > 6:  # Only sample when many empty cells
            sample_size = max(3, min(6, int(len(empty_cells) * self.sampling_ratio)))

            # Prioritize cells by strategic importance
            def cell_priority(cell):
                i, j = cell
                # Snake pattern priority (following position weights)
                priority = self.position_weights[i, j]

                # Bonus for cells adjacent to high-value tiles
                adjacent_bonus = 0
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4 and game_state.grid[ni, nj] > 0:
                        adjacent_bonus += np.log2(game_state.grid[ni, nj])

                return priority + adjacent_bonus * 0.5

            # Sort cells by priority and take top samples
            sorted_cells = sorted(empty_cells, key=cell_priority, reverse=True)
            sampled_cells = sorted_cells[:sample_size]
        else:
            sampled_cells = empty_cells

        expected_score = 0.0

        # Compute expected value over all sampled positions
        for i, j in sampled_cells:
            # Try spawning a 2 tile (90% probability)
            test_game = self.copy_game_state(game_state)
            test_game.grid[i, j] = 2
            score_2 = self._expectimax_max_node(test_game, depth - 1)

            # Try spawning a 4 tile (10% probability)
            test_game.grid[i, j] = 4
            score_4 = self._expectimax_max_node(test_game, depth - 1)

            # Weighted average for this position
            expected_score += 0.9 * score_2 + 0.1 * score_4

        # Average over all sampled positions
        return expected_score / len(sampled_cells)

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
            return -200000  # Heavily penalize game over

        # Quick win condition bonus
        if np.max(grid) >= 2048:
            return 200000

        score = 0.0

        # Calculate all heuristics
        monotonicity = self._monotonicity(grid)
        smoothness = self._smoothness(grid)
        free_tiles = self._free_tiles(grid)
        max_tile_corner = self._max_tile_corner(grid)
        merge_potential = self._merge_potential(grid)
        weighted_positions = self._weighted_positions(grid)
        score_bonus = np.sum(grid)  # Actual tile values
        tile_clustering = self._tile_clustering_penalty(grid)

        # Combine with weights
        score += self.weights["monotonicity"] * monotonicity
        score += self.weights["smoothness"] * smoothness
        score += self.weights["free_tiles"] * free_tiles
        score += self.weights["max_tile_corner"] * max_tile_corner
        score += self.weights["merge_potential"] * merge_potential
        score += self.weights["weighted_positions"] * weighted_positions
        score += self.weights["score_bonus"] * score_bonus
        score += self.weights["tile_clustering"] * tile_clustering

        # Exponential bonus for high-value tiles (increased)
        max_tile = np.max(grid)
        if max_tile > 0:
            score += np.log2(max_tile) * 300  # Doubled bonus

        # Additional bonuses for tile progression
        unique_tiles = len(np.unique(grid[grid > 0]))
        score += unique_tiles * 50  # Reward tile diversity

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
            return np.log2(max_tile) * 3

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
            return np.log2(max_tile) * 1.5

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

    def _tile_clustering_penalty(self, grid: np.ndarray) -> float:
        """Penalize scattered high-value tiles."""
        max_tile = np.max(grid)

        if max_tile <= 4:
            return 0.0

        # Find positions of high-value tiles (>= max_tile/4)
        threshold = max_tile // 4
        high_tiles = []

        for i in range(4):
            for j in range(4):
                if grid[i, j] >= threshold:
                    high_tiles.append((i, j, grid[i, j]))

        # Calculate average distance between high tiles
        if len(high_tiles) <= 1:
            return 0.0

        total_distance = 0.0
        pairs = 0

        for i in range(len(high_tiles)):
            for j in range(i + 1, len(high_tiles)):
                pos1, pos2 = high_tiles[i][:2], high_tiles[j][:2]
                distance = abs(pos1[0] - pos2[0]) + abs(
                    pos1[1] - pos2[1]
                )  # Manhattan distance
                total_distance += distance
                pairs += 1

        avg_distance = total_distance / pairs if pairs > 0 else 0
        return avg_distance * 10  # Return penalty (higher distance = higher penalty)

    def set_depth(self, depth: int) -> None:
        """Set the maximum search depth."""
        self.max_depth = max(1, depth)

    def set_sampling_ratio(self, ratio: float) -> None:
        """Set the chance node sampling ratio."""
        self.sampling_ratio = max(0.1, min(1.0, ratio))
