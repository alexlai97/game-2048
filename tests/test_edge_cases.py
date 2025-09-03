import pytest
import numpy as np
from game_2048.core import Game2048


class TestEdgeCases:
    """
    Comprehensive test suite for edge cases in the 2048 game.
    Tests boundary conditions, invalid inputs, and stress scenarios.
    """
    
    # ========== FULL BOARD SCENARIOS ==========
    
    def test_completely_full_board_no_merges_game_over(self):
        """Test board completely full with no possible merges (true game over)."""
        game = Game2048()
        
        # Create a checkerboard pattern with no possible merges
        game.grid = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        
        assert game.is_game_over()
        assert game.get_state() == "lost"
        assert len(game.get_available_moves()) == 0
        
        # Verify no moves work
        for direction in ['left', 'right', 'up', 'down']:
            assert not game.move(direction)
    
    def test_full_board_with_possible_merges(self):
        """Test board full but with merges still available."""
        game = Game2048()
        
        # Full board but with mergeable tiles
        game.grid = np.array([
            [2, 2, 4, 8],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128]
        ])
        
        assert not game.is_game_over()
        assert game.get_state() == "ongoing"
        available_moves = game.get_available_moves()
        assert len(available_moves) > 0
        
        # Should be able to make at least one move
        original_score = game.score
        moved = game.move('left')
        assert moved
        assert game.score > original_score
    
    def test_board_with_one_empty_cell(self):
        """Test board with only one empty cell remaining."""
        game = Game2048()
        
        # Fill all but one cell
        game.grid = np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 0]  # One empty cell
        ])
        
        # Should still be able to move if merges are possible
        assert not game.is_game_over()
        
        # Make a move that doesn't create merges
        original_grid = game.grid.copy()
        game.move('down')  # Should just move tiles down and spawn in empty cell
        
        # Board should now be full
        assert np.sum(game.grid == 0) == 0
    
    def test_full_board_edge_case_single_merge_available(self):
        """Test full board where only one merge is possible in one direction."""
        game = Game2048()
        
        game.grid = np.array([
            [2, 2, 4, 8],     # Only these 2s can merge (left or right)
            [4, 8, 16, 32], 
            [8, 16, 32, 64],
            [16, 32, 64, 128]
        ])
        
        available_moves = game.get_available_moves()
        assert len(available_moves) == 2  # left and right both work
        assert 'left' in available_moves and 'right' in available_moves
    
    # ========== MOVEMENT EDGE CASES ==========
    
    def test_movement_on_completely_full_board(self):
        """Test attempting moves when board is completely full with no merges."""
        game = Game2048()
        
        # Create unmergeable full board
        game.grid = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        
        original_grid = game.grid.copy()
        original_score = game.score
        
        # All moves should fail
        for direction in ['left', 'right', 'up', 'down']:
            result = game.move(direction)
            assert not result
            np.testing.assert_array_equal(game.grid, original_grid)
            assert game.score == original_score
    
    def test_rapid_consecutive_moves(self):
        """Test rapid consecutive moves for state consistency."""
        game = Game2048()
        
        # Set up a specific board state
        game.grid = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 4]
        ])
        
        # Make rapid consecutive moves
        moves = ['left', 'up', 'right', 'down', 'left']
        for move in moves:
            original_grid = game.grid.copy()
            moved = game.move(move)
            
            # Ensure grid is always valid
            assert game.grid.shape == (4, 4)
            assert np.all(game.grid >= 0)
            assert np.all((game.grid == 0) | (game.grid >= 2))
            
            # Ensure scores are monotonically increasing or staying same
            assert game.score >= 0
    
    def test_moving_into_wall_no_change(self):
        """Test moving tiles that are already against a wall."""
        game = Game2048()
        
        # Tiles already at left edge
        game.grid = np.array([
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0]
        ])
        
        original_grid = game.grid.copy()
        original_score = game.score
        
        # Moving left should not change anything
        moved = game.move_left()
        assert not moved
        np.testing.assert_array_equal(game.grid, original_grid)
        assert game.score == original_score
    
    def test_no_movement_possible_in_direction(self):
        """Test scenarios where movement in specific directions is impossible."""
        game = Game2048()
        
        # Create scenario where only up/down movement is possible
        game.grid = np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        # Left and right should not work (no merges, already compacted)
        original_grid = game.grid.copy()
        
        assert not game.move_left()
        assert not game.move_right()
        np.testing.assert_array_equal(game.grid, original_grid)
        
        # But up and down should work
        assert game.move_up() or game.move_down()
    
    # ========== BOUNDARY CONDITIONS ==========
    
    def test_corner_tiles_movement(self):
        """Test tiles at corners moving in various directions."""
        game = Game2048()
        
        # Place tiles only in corners
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 2  # Top-left
        game.grid[0, 3] = 4  # Top-right
        game.grid[3, 0] = 8  # Bottom-left
        game.grid[3, 3] = 16 # Bottom-right
        
        # Test each direction
        original_grid = game.grid.copy()
        
        # Moving left: top-right and bottom-right should move
        game.move_left()
        assert game.grid[0, 0] == 2  # Should stay
        assert game.grid[0, 1] == 4  # Should move from [0,3]
        assert game.grid[3, 0] == 8  # Should stay
        assert game.grid[3, 1] == 16 # Should move from [3,3]
    
    def test_single_tile_on_empty_board(self):
        """Test single tile on otherwise empty board."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        game.grid[2, 2] = 2  # Single tile in center
        
        # Each direction should move the tile to the corresponding edge
        test_directions = [
            ('left', (2, 0)),
            ('right', (2, 3)),
            ('up', (0, 2)),
            ('down', (3, 2))
        ]
        
        for direction, expected_pos in test_directions:
            test_game = Game2048()
            test_game.grid = np.zeros((4, 4))
            test_game.grid[2, 2] = 2
            
            getattr(test_game, f'move_{direction}')()
            assert test_game.grid[expected_pos] == 2
            assert np.sum(test_game.grid > 0) == 1  # Should still be only one tile
    
    def test_maximum_value_tiles(self):
        """Test behavior with maximum value tiles (2048, 4096, etc.)."""
        game = Game2048()
        
        # Test with 2048 (winning tile)
        game.grid = np.array([
            [2048, 2048, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        original_score = game.score
        game.move_left()
        
        # Should merge to 4096
        assert game.grid[0, 0] == 4096
        assert game.score == original_score + 4096
        assert game.has_won()  # Should still be won state
    
    def test_extremely_high_value_tiles(self):
        """Test with unrealistically high value tiles for overflow testing."""
        game = Game2048()
        
        # Test with very high values
        high_value = 32768  # 2^15
        game.grid = np.array([
            [high_value, high_value, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        original_score = game.score
        game.move_left()
        
        # Should merge to double the value
        assert game.grid[0, 0] == high_value * 2
        assert game.score == original_score + (high_value * 2)
    
    # ========== INVALID INPUT HANDLING ==========
    
    def test_invalid_direction_strings(self):
        """Test handling of invalid direction strings."""
        game = Game2048()
        
        invalid_directions = [
            'invalid', 'LEFT', 'Right', 'UP', 'down_arrow',
            '', ' ', 'north', 'south', 'east', 'west',
            '1', 'left ', ' right', 'up\n', 'down\t'
        ]
        
        for direction in invalid_directions:
            original_grid = game.grid.copy()
            result = game.move(direction)
            assert not result
            np.testing.assert_array_equal(game.grid, original_grid)
    
    def test_case_sensitivity_directions(self):
        """Test that direction inputs are case sensitive."""
        game = Game2048()
        original_grid = game.grid.copy()
        
        case_variations = ['LEFT', 'Right', 'UP', 'Down', 'LeFt', 'rIgHt']
        
        for direction in case_variations:
            result = game.move(direction)
            assert not result
            np.testing.assert_array_equal(game.grid, original_grid)
    
    def test_none_and_empty_direction(self):
        """Test handling of None and empty direction inputs."""
        game = Game2048()
        original_grid = game.grid.copy()
        
        # Test None (should return False gracefully)
        result = game.move(None)
        assert not result
        np.testing.assert_array_equal(game.grid, original_grid)
        
        # Test empty string
        result = game.move('')
        assert not result
        np.testing.assert_array_equal(game.grid, original_grid)
    
    # ========== MEMORY AND STATE CONSISTENCY ==========
    
    def test_grid_integrity_after_operations(self):
        """Test that grid maintains integrity after various operations."""
        game = Game2048()
        
        # Perform various operations
        operations = ['left', 'right', 'up', 'down'] * 10
        
        for operation in operations:
            game.move(operation)
            
            # Check grid integrity
            assert game.grid.shape == (4, 4)
            assert game.grid.dtype == int
            assert np.all(game.grid >= 0)
            
            # Check that non-zero values are powers of 2
            non_zero_values = game.grid[game.grid > 0]
            for value in non_zero_values:
                assert value >= 2
                # Check if power of 2
                assert (value & (value - 1)) == 0
    
    def test_score_consistency(self):
        """Test that score updates are consistent and monotonic."""
        game = Game2048()
        
        # Set up a board with guaranteed merges
        game.grid = np.array([
            [2, 2, 4, 4],
            [2, 2, 4, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        original_score = game.score
        game.move_left()
        
        # Score should increase by the sum of merged values
        expected_increase = 4 + 8 + 4 + 8  # Two 4s and two 8s created
        assert game.score == original_score + expected_increase
        
        # Score should never decrease
        for _ in range(20):
            prev_score = game.score
            game.move('left')
            assert game.score >= prev_score
    
    def test_no_state_corruption_after_failed_moves(self):
        """Test that failed moves don't corrupt game state."""
        game = Game2048()
        
        # Create a state where no moves are possible
        game.grid = np.array([
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2]
        ])
        
        original_grid = game.grid.copy()
        original_score = game.score
        
        # Attempt many failed moves
        for _ in range(100):
            for direction in ['left', 'right', 'up', 'down']:
                game.move(direction)
        
        # State should be unchanged
        np.testing.assert_array_equal(game.grid, original_grid)
        assert game.score == original_score
        assert game.is_game_over()
    
    def test_grid_copy_independence(self):
        """Test that grid copies are truly independent."""
        game = Game2048()
        copy1 = game.get_grid_copy()
        copy2 = game.get_grid_copy()
        
        # Modify one copy
        copy1[0, 0] = 9999
        
        # Other copy and original should be unchanged
        assert game.grid[0, 0] != 9999
        assert copy2[0, 0] != 9999
        assert copy1 is not copy2
        assert copy1 is not game.grid
        assert copy2 is not game.grid
    
    # ========== PERFORMANCE EDGE CASES ==========
    
    def test_large_value_computations(self):
        """Test computations with large tile values."""
        game = Game2048()
        
        # Set extremely high values
        max_value = 131072  # 2^17
        game.grid = np.array([
            [max_value, max_value, 0, 0],
            [max_value//2, max_value//2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        # Perform merges
        game.move_left()
        
        # Check that computations are correct
        assert game.grid[0, 0] == max_value * 2
        assert game.grid[1, 0] == max_value
        
        # Score should be calculated correctly
        expected_score_increase = (max_value * 2) + max_value
        assert game.score == expected_score_increase
    
    def test_repeated_operations_performance(self):
        """Test repeated operations for performance and consistency."""
        game = Game2048()
        
        # Perform many operations
        for i in range(1000):
            direction = ['left', 'right', 'up', 'down'][i % 4]
            
            prev_state = {
                'grid': game.grid.copy(),
                'score': game.score
            }
            
            moved = game.move(direction)
            
            # Ensure state is always valid
            assert game.grid.shape == (4, 4)
            assert game.score >= prev_state['score']
            
            if not moved:
                # If no move occurred, state should be unchanged
                np.testing.assert_array_equal(game.grid, prev_state['grid'])
                assert game.score == prev_state['score']
    
    def test_stress_test_random_operations(self):
        """Stress test with random operations."""
        import random
        
        for _ in range(100):  # Run 100 different games
            game = Game2048()
            
            # Play random game
            for _ in range(200):  # Up to 200 moves per game
                if game.is_game_over():
                    break
                
                direction = random.choice(['left', 'right', 'up', 'down'])
                game.move(direction)
                
                # Invariants that must always hold
                assert game.grid.shape == (4, 4)
                assert np.all(game.grid >= 0)
                assert game.score >= 0
                
                # Check power of 2 constraint
                non_zero = game.grid[game.grid > 0]
                for value in non_zero:
                    assert value >= 2 and (value & (value - 1)) == 0
    
    # ========== BOUNDARY VALUE TESTING ==========
    
    def test_minimal_winning_condition(self):
        """Test minimal condition for winning (exactly 2048)."""
        game = Game2048()
        
        # Set up board with exactly 2048
        game.grid = np.zeros((4, 4))
        game.grid[0, 0] = 2048
        
        assert game.has_won()
        assert game.get_state() == "won"
    
    def test_just_below_winning_condition(self):
        """Test condition just below winning (1024 + 1024)."""
        game = Game2048()
        
        # Set up board that can reach 2048 in one move
        game.grid = np.array([
            [1024, 1024, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        assert not game.has_won()
        assert game.get_state() == "ongoing"
        
        # Make the winning move
        game.move_left()
        
        assert game.has_won()
        assert game.get_state() == "won"
    
    def test_empty_board_edge_case(self):
        """Test behavior with manually emptied board."""
        game = Game2048()
        game.grid = np.zeros((4, 4))
        
        # Empty board should have no available moves (nothing to move)
        available_moves = game.get_available_moves()
        
        # No moves should be available on empty board
        assert len(available_moves) == 0
        
        # Attempting any move should return False  
        assert not game.move('left')
        assert not game.move('right')
        assert not game.move('up')
        assert not game.move('down')
        
        # Board should remain empty
        assert np.sum(game.grid > 0) == 0