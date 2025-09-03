import numpy as np

from game_2048.core import Game2048


class TestGameStates:
    """Comprehensive tests for game state conditions (win/lose) in 2048 game."""

    def test_initial_state_ongoing(self):
        """Test that a new game starts in ongoing state."""
        game = Game2048()
        assert game.get_state() == "ongoing"
        assert not game.has_won()
        assert not game.is_game_over()

    # ========== WIN CONDITIONS ==========

    def test_win_condition_exactly_2048(self):
        """Test winning when exactly 2048 is reached."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 2048
        game.grid[0, 1] = 2

        assert game.has_won()
        assert game.get_state() == "won"
        assert not game.is_game_over()

    def test_win_condition_multiple_2048_tiles(self):
        """Test winning with multiple 2048 tiles."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 2048
        game.grid[1, 1] = 2048
        game.grid[2, 2] = 4

        assert game.has_won()
        assert game.get_state() == "won"

    def test_win_condition_higher_values_4096(self):
        """Test winning with tiles higher than 2048 (4096)."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 4096
        game.grid[1, 0] = 2

        assert game.has_won()
        assert game.get_state() == "won"

    def test_win_condition_higher_values_8192(self):
        """Test winning with tiles higher than 2048 (8192)."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 8192
        game.grid[1, 0] = 2

        assert game.has_won()
        assert game.get_state() == "won"

    def test_win_condition_maximum_theoretical_value(self):
        """Test winning with maximum theoretical value (131072)."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 131072  # 2^17, theoretical maximum
        game.grid[1, 0] = 2

        assert game.has_won()
        assert game.get_state() == "won"

    def test_continue_after_win(self):
        """Test that game can continue after winning (2048 reached)."""
        game = Game2048()
        game.grid = np.array(
            [[2048, 2, 0, 0], [4, 8, 0, 0], [2, 4, 0, 0], [0, 0, 0, 0]]
        )

        # Game should be won but still playable
        assert game.has_won()
        assert game.get_state() == "won"
        assert not game.is_game_over()
        assert len(game.get_available_moves()) > 0

    # ========== LOSS CONDITIONS ==========

    def test_loss_condition_no_moves_available(self):
        """Test losing when no valid moves are available."""
        game = Game2048()
        # Create a board with no possible merges and no empty spaces
        game.grid = np.array(
            [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
        )

        assert game.is_game_over()
        assert game.get_state() == "lost"
        assert len(game.get_available_moves()) == 0

    def test_loss_condition_full_board_no_merges(self):
        """Test losing with a completely full board and no possible merges."""
        game = Game2048()
        game.grid = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])

        assert game.is_game_over()
        assert game.get_state() == "lost"
        assert not game.has_won()

    def test_loss_condition_checkerboard_pattern(self):
        """Test losing with alternating pattern that prevents merges."""
        game = Game2048()
        game.grid = np.array(
            [[8, 16, 8, 16], [16, 8, 16, 8], [8, 16, 8, 16], [16, 8, 16, 8]]
        )

        assert game.is_game_over()
        assert game.get_state() == "lost"

    def test_loss_condition_high_values_no_moves(self):
        """Test losing with high values but no available moves."""
        game = Game2048()
        game.grid = np.array(
            [
                [1024, 512, 256, 128],
                [512, 256, 128, 64],
                [256, 128, 64, 32],
                [128, 64, 32, 16],
            ]
        )

        assert game.is_game_over()
        assert game.get_state() == "lost"

    # ========== ONGOING STATE ==========

    def test_ongoing_state_with_empty_cells(self):
        """Test ongoing state with empty cells available."""
        game = Game2048()
        game.grid = np.array([[2, 4, 8, 0], [4, 8, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0]])

        assert game.get_state() == "ongoing"
        assert not game.has_won()
        assert not game.is_game_over()
        assert len(game.get_available_moves()) > 0

    def test_ongoing_state_with_possible_merges(self):
        """Test ongoing state with possible merges but no empty cells."""
        game = Game2048()
        game.grid = np.array(
            [[2, 2, 4, 8], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
        )

        assert game.get_state() == "ongoing"
        assert not game.has_won()
        assert not game.is_game_over()
        assert len(game.get_available_moves()) > 0

    def test_ongoing_state_full_board_with_merges(self):
        """Test ongoing state with full board but merge possibilities."""
        game = Game2048()
        game.grid = np.array(
            [
                [2, 4, 8, 16],
                [2, 8, 16, 32],  # Vertical merge possible: 2,2
                [4, 16, 32, 64],
                [8, 32, 64, 128],
            ]
        )

        assert game.get_state() == "ongoing"
        assert not game.is_game_over()
        assert len(game.get_available_moves()) > 0

    # ========== STATE TRANSITIONS ==========

    def test_transition_ongoing_to_won(self):
        """Test transition from ongoing to won state."""
        game = Game2048()
        game.grid = np.array(
            [[1024, 1024, 2, 0], [4, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        # Initial state should be ongoing
        assert game.get_state() == "ongoing"

        # Simulate move that creates 2048
        game.move_left()

        # Should now be won
        assert game.has_won()
        assert game.get_state() == "won"

    def test_transition_ongoing_to_lost(self):
        """Test transition from ongoing to lost state."""
        game = Game2048()
        # Set up a nearly lost position with one possible move
        game.grid = np.array(
            [
                [2, 4, 8, 16],
                [4, 8, 16, 32],
                [8, 16, 32, 64],
                [16, 32, 64, 0],  # One empty space
            ]
        )

        # Should be ongoing with one move available
        assert game.get_state() == "ongoing"
        assert not game.is_game_over()

        # Fill the last space to create game over
        game.grid[3, 3] = 128

        # Should now be lost
        assert game.is_game_over()
        assert game.get_state() == "lost"

    def test_won_state_persists_after_moves(self):
        """Test that won state persists even after making moves."""
        game = Game2048()
        game.grid = np.array(
            [[2048, 2, 4, 0], [4, 8, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        # Should be won
        assert game.get_state() == "won"

        # Make a move
        game.move_left()

        # Should still be won
        assert game.get_state() == "won"
        assert game.has_won()

    # ========== EDGE CASES ==========

    def test_win_and_lose_impossible_simultaneously(self):
        """Test that win and lose cannot occur simultaneously."""
        game = Game2048()
        # Create winning board that's also full
        game.grid = np.array(
            [[2048, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
        )

        # Win condition takes precedence
        assert game.has_won()
        assert game.get_state() == "won"
        # Even though technically no moves available, win takes precedence

    def test_state_consistency_empty_board(self):
        """Test state consistency with mostly empty board."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 2

        assert game.get_state() == "ongoing"
        assert not game.has_won()
        assert not game.is_game_over()

    def test_state_consistency_single_tile_2048(self):
        """Test state consistency with only a 2048 tile."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 2048

        assert game.get_state() == "won"
        assert game.has_won()
        assert not game.is_game_over()

    def test_edge_case_maximum_board_values(self):
        """Test with maximum possible values on board."""
        game = Game2048()
        # Fill with very high values
        game.grid = np.array(
            [
                [131072, 65536, 32768, 16384],
                [65536, 32768, 16384, 8192],
                [32768, 16384, 8192, 4096],
                [16384, 8192, 4096, 2048],
            ]
        )

        assert game.has_won()
        assert game.get_state() == "won"
        # Should still be able to detect available moves if any exist

    def test_boundary_values_just_below_2048(self):
        """Test with values just below 2048."""
        game = Game2048()
        game.grid = np.array(
            [[1024, 1024, 4, 0], [512, 256, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        assert not game.has_won()
        assert game.get_state() == "ongoing"
        assert not game.is_game_over()

    def test_get_available_moves_consistency_with_game_over(self):
        """Test that get_available_moves() is consistent with is_game_over()."""
        game = Game2048()
        game.grid = np.array(
            [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
        )

        available_moves = game.get_available_moves()
        is_over = game.is_game_over()

        # These should be consistent
        assert (len(available_moves) == 0) == is_over

    def test_state_detection_performance_full_board(self):
        """Test state detection with various full board configurations."""
        game = Game2048()

        # Test multiple full board scenarios
        test_boards = [
            # Completely blocked board
            np.array(
                [[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
            ),
            # Board with horizontal merge possibility
            np.array(
                [[2, 2, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
            ),
            # Board with vertical merge possibility
            np.array(
                [[2, 4, 8, 16], [2, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]]
            ),
        ]

        expected_results = [True, False, False]  # is_game_over results

        for i, board in enumerate(test_boards):
            game.grid = board
            assert game.is_game_over() == expected_results[i], f"Board {i} failed"

    def test_state_methods_return_types(self):
        """Test that state methods return correct types."""
        game = Game2048()

        # Test return types
        assert isinstance(game.get_state(), str)
        # numpy boolean types are also valid boolean types
        assert isinstance(game.has_won(), (bool, np.bool_))
        assert isinstance(game.is_game_over(), (bool, np.bool_))
        assert isinstance(game.get_available_moves(), list)

        # Test valid state values
        state = game.get_state()
        assert state in ["ongoing", "won", "lost"]
