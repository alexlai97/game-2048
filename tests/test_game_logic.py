import numpy as np

from game_2048.core import Game2048


class TestGameLogic:
    def test_initialization(self):
        game = Game2048()
        assert game.score == 0
        assert game.grid.shape == (4, 4)
        assert np.sum(game.grid > 0) == 2  # Two initial tiles
        assert np.all((game.grid == 0) | (game.grid == 2) | (game.grid == 4))

    def test_spawn_tile(self):
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game._spawn_tile()
        assert np.sum(game.grid > 0) == 1
        assert game.grid[game.grid > 0][0] in [2, 4]

    def test_spawn_tile_probability(self):
        # Test that spawning follows 90% 2s, 10% 4s distribution
        game = Game2048()
        spawn_values = []

        for _ in range(100):
            game.grid = np.zeros((4, 4))
            game._spawn_tile()
            spawn_values.append(game.grid[game.grid > 0][0])

        twos = spawn_values.count(2)
        fours = spawn_values.count(4)

        # Allow some variance but check rough distribution
        assert 80 <= twos <= 100  # Should be around 90
        assert 0 <= fours <= 20  # Should be around 10

    def test_no_spawn_on_full_board(self):
        game = Game2048()
        game.grid = np.full((4, 4), 2)  # Fill board

        original_grid = game.grid.copy()
        game._spawn_tile()

        # Grid should remain unchanged
        np.testing.assert_array_equal(game.grid, original_grid)

    def test_slide_line_basic(self):
        game = Game2048()

        # Test sliding with gaps
        line = np.array([2, 0, 0, 2])
        result, score = game._slide_line(line)
        expected = np.array([4, 0, 0, 0])
        np.testing.assert_array_equal(result, expected)
        assert score == 4

    def test_slide_line_no_merge(self):
        game = Game2048()

        # Test sliding without merging
        line = np.array([2, 4, 8, 0])
        result, score = game._slide_line(line)
        expected = np.array([2, 4, 8, 0])
        np.testing.assert_array_equal(result, expected)
        assert score == 0

    def test_slide_line_multiple_merges(self):
        game = Game2048()

        # Test multiple merges
        line = np.array([2, 2, 4, 4])
        result, score = game._slide_line(line)
        expected = np.array([4, 8, 0, 0])
        np.testing.assert_array_equal(result, expected)
        assert score == 12  # 4 + 8

    def test_slide_line_no_triple_merge(self):
        game = Game2048()

        # Test that [4,4,4,0] becomes [8,4,0,0] not [16,0,0,0]
        line = np.array([4, 4, 4, 0])
        result, score = game._slide_line(line)
        expected = np.array([8, 4, 0, 0])
        np.testing.assert_array_equal(result, expected)
        assert score == 8

    def test_get_available_moves(self):
        game = Game2048()

        # Test with empty board (except 2 initial tiles)
        moves = game.get_available_moves()
        assert len(moves) > 0
        assert all(move in ["left", "right", "up", "down"] for move in moves)

    def test_get_state(self):
        game = Game2048()

        # Initial state should be ongoing
        assert game.get_state() == "ongoing"

        # Test win state
        game.grid[0, 0] = 2048
        assert game.get_state() == "won"

    def test_grid_copy(self):
        game = Game2048()
        copy = game.get_grid_copy()

        # Should be equal but different objects
        np.testing.assert_array_equal(game.grid, copy)
        assert game.grid is not copy
