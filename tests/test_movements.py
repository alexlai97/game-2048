import pytest
import numpy as np
from game_2048.core import Game2048


class TestMovements:
    """Comprehensive tests for all movement directions in 2048 game."""
    
    @pytest.fixture
    def empty_game(self):
        """Create a game with an empty grid for testing."""
        game = Game2048()
        game.grid = np.zeros((4, 4), dtype=int)
        game.score = 0
        return game
    
    # Test basic sliding in each direction
    @pytest.mark.parametrize("direction,initial_grid,expected_grid", [
        # Left movement tests
        ("left", 
         [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]], 
         [[2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]]),
        ("left", 
         [[2, 2, 0, 0], [4, 0, 4, 0], [2, 4, 2, 8], [0, 0, 0, 0]], 
         [[4, 0, 0, 0], [8, 0, 0, 0], [2, 4, 2, 8], [0, 0, 0, 0]]),
        
        # Right movement tests  
        ("right", 
         [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]], 
         [[0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 2], [0, 0, 0, 2]]),
        ("right", 
         [[2, 2, 0, 0], [4, 0, 4, 0], [2, 4, 2, 8], [0, 0, 0, 0]], 
         [[0, 0, 0, 4], [0, 0, 0, 8], [2, 4, 2, 8], [0, 0, 0, 0]]),
        
        # Up movement tests
        ("up", 
         [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]], 
         [[2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("up", 
         [[2, 4, 2, 0], [2, 0, 4, 0], [0, 4, 2, 0], [0, 0, 8, 0]], 
         [[4, 8, 2, 0], [0, 0, 4, 0], [0, 0, 2, 0], [0, 0, 8, 0]]),
        
        # Down movement tests
        ("down", 
         [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]], 
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 2, 2, 2]]),
        ("down", 
         [[2, 4, 2, 0], [2, 0, 4, 0], [0, 4, 2, 0], [0, 0, 8, 0]], 
         [[0, 0, 2, 0], [0, 0, 4, 0], [0, 0, 2, 0], [4, 8, 8, 0]]),
    ])
    def test_basic_movements(self, empty_game, direction, initial_grid, expected_grid):
        """Test basic tile sliding in each direction."""
        empty_game.grid = np.array(initial_grid, dtype=int)
        
        # Perform the movement
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        # Check the result
        np.testing.assert_array_equal(empty_game.grid, np.array(expected_grid, dtype=int))
        assert result == True  # Movement should be valid
    
    # Test movement with gaps between tiles
    @pytest.mark.parametrize("direction,initial_grid,expected_grid", [
        # Left - tiles with gaps
        ("left", 
         [[2, 0, 0, 2], [0, 4, 0, 4], [8, 0, 8, 0], [0, 2, 0, 2]], 
         [[4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0], [4, 0, 0, 0]]),
        
        # Right - tiles with gaps
        ("right", 
         [[2, 0, 0, 2], [0, 4, 0, 4], [8, 0, 8, 0], [0, 2, 0, 2]], 
         [[0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16], [0, 0, 0, 4]]),
        
        # Up - tiles with gaps
        ("up", 
         [[2, 0, 8, 0], [0, 4, 0, 2], [0, 0, 8, 0], [2, 4, 0, 2]], 
         [[4, 8, 16, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        
        # Down - tiles with gaps
        ("down", 
         [[2, 0, 8, 0], [0, 4, 0, 2], [0, 0, 8, 0], [2, 4, 0, 2]], 
         [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 8, 16, 4]]),
    ])
    def test_movement_with_gaps(self, empty_game, direction, initial_grid, expected_grid):
        """Test that tiles slide correctly across gaps."""
        empty_game.grid = np.array(initial_grid, dtype=int)
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        np.testing.assert_array_equal(empty_game.grid, np.array(expected_grid, dtype=int))
        assert result == True
    
    # Test invalid moves (no movement possible)
    @pytest.mark.parametrize("direction,grid_state", [
        # Left - all tiles already at left
        ("left", [[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]]),
        
        # Right - all tiles already at right
        ("right", [[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 0, 16]]),
        
        # Up - all tiles already at top
        ("up", [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        
        # Down - all tiles already at bottom
        ("down", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 4, 8, 16]]),
        
        # No possible merges and no movement
        ("left", [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 64]]),
        ("right", [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 64]]),
        ("up", [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 64]]),
        ("down", [[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2, 4], [8, 16, 32, 64]]),
    ])
    def test_invalid_moves(self, empty_game, direction, grid_state):
        """Test that invalid moves return False and don't change the grid."""
        empty_game.grid = np.array(grid_state, dtype=int)
        original_grid = empty_game.grid.copy()
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        # Move should be invalid
        assert result == False
        # Grid should remain unchanged
        np.testing.assert_array_equal(empty_game.grid, original_grid)
    
    # Test that valid moves spawn new tiles when using the move() method
    @pytest.mark.parametrize("direction,setup_grid", [
        ("left", [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # Can move left
        ("right", [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),  # Can move right  
        ("up", [[0, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),    # Can move up
        ("down", [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])   # Can move down
    ])
    def test_valid_moves_spawn_tiles(self, empty_game, direction, setup_grid):
        """Test that valid moves return True and spawn new tiles."""
        # Set up a grid where movement in the specific direction is possible
        empty_game.grid = np.array(setup_grid, dtype=int)
        
        original_tile_count = np.sum(empty_game.grid > 0)
        
        # Use the main move method which spawns tiles
        result = empty_game.move(direction)
        
        # Move should be valid
        assert result == True
        # Should have one more tile than before
        new_tile_count = np.sum(empty_game.grid > 0)
        assert new_tile_count == original_tile_count + 1
    
    # Test the regression case: move_down with tiles at the top
    def test_move_down_regression_case(self, empty_game):
        """Test the specific regression case that was fixed: move_down with tiles at top."""
        # Set up tiles at the top of the board
        empty_game.grid = np.array([
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=int)
        
        result = empty_game.move_down()
        
        # All tiles should move to the bottom
        expected_grid = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 4, 8, 16]
        ], dtype=int)
        
        assert result == True
        np.testing.assert_array_equal(empty_game.grid, expected_grid)
    
    # Test edge case: tiles at edges trying to move further
    @pytest.mark.parametrize("direction,initial_grid", [
        ("left", [[2, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("right", [[0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("up", [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        ("down", [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 4, 0, 0]]),
    ])
    def test_tiles_at_edges(self, empty_game, direction, initial_grid):
        """Test behavior when tiles are already at the edge they're trying to move to."""
        empty_game.grid = np.array(initial_grid, dtype=int)
        original_grid = empty_game.grid.copy()
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        # Should return False since no movement is possible
        assert result == False
        # Grid should remain unchanged
        np.testing.assert_array_equal(empty_game.grid, original_grid)
    
    # Test complex merging scenarios
    @pytest.mark.parametrize("direction,initial_grid,expected_grid,expected_score", [
        # Left - complex merging with multiple pairs
        ("left", 
         [[2, 2, 4, 4], [8, 8, 16, 16], [2, 0, 2, 0], [4, 8, 4, 8]], 
         [[4, 8, 0, 0], [16, 32, 0, 0], [4, 0, 0, 0], [4, 8, 4, 8]], 
         4 + 8 + 16 + 32 + 4),  # Score from merges
        
        # Right - complex merging
        ("right", 
         [[2, 2, 4, 4], [8, 8, 16, 16], [0, 2, 0, 2], [4, 8, 4, 8]], 
         [[0, 0, 4, 8], [0, 0, 16, 32], [0, 0, 0, 4], [4, 8, 4, 8]], 
         4 + 8 + 16 + 32 + 4),
        
        # Up - complex merging
        ("up", 
         [[2, 8, 0, 4], [2, 8, 2, 8], [4, 16, 0, 4], [4, 16, 2, 8]], 
         [[4, 16, 4, 4], [8, 32, 0, 8], [0, 0, 0, 4], [0, 0, 0, 8]], 
         64),
        
        # Down - complex merging
        ("down", 
         [[2, 8, 2, 4], [2, 8, 0, 8], [4, 16, 0, 4], [4, 16, 2, 8]], 
         [[0, 0, 0, 4], [0, 0, 0, 8], [4, 16, 0, 4], [8, 32, 4, 8]], 
         64),
    ])
    def test_complex_merging(self, empty_game, direction, initial_grid, expected_grid, expected_score):
        """Test complex merging scenarios with score calculation."""
        empty_game.grid = np.array(initial_grid, dtype=int)
        initial_score = empty_game.score
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        assert result == True
        np.testing.assert_array_equal(empty_game.grid, np.array(expected_grid, dtype=int))
        assert empty_game.score == initial_score + expected_score
    
    # Test that no triple merging occurs
    @pytest.mark.parametrize("direction,initial_grid,expected_grid", [
        ("left", [[4, 4, 4, 0]], [[8, 4, 0, 0]]),
        ("right", [[0, 4, 4, 4]], [[0, 0, 4, 8]]),
        ("up", [[4], [4], [4], [0]], [[8], [4], [0], [0]]),
        ("down", [[0], [4], [4], [4]], [[0], [0], [4], [8]]),
    ])
    def test_no_triple_merging(self, empty_game, direction, initial_grid, expected_grid):
        """Test that three identical tiles don't all merge into one."""
        # Convert single row/column tests to full 4x4 grids
        if direction in ["left", "right"]:
            full_initial = np.zeros((4, 4), dtype=int)
            full_expected = np.zeros((4, 4), dtype=int)
            full_initial[0] = initial_grid[0]
            full_expected[0] = expected_grid[0]
        else:  # up, down
            full_initial = np.zeros((4, 4), dtype=int)
            full_expected = np.zeros((4, 4), dtype=int)
            for i in range(4):
                full_initial[i][0] = initial_grid[i][0]
                full_expected[i][0] = expected_grid[i][0]
        
        empty_game.grid = full_initial
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        assert result == True
        np.testing.assert_array_equal(empty_game.grid, full_expected)
    
    # Test empty grid (no tiles to move)
    @pytest.mark.parametrize("direction", ["left", "right", "up", "down"])
    def test_empty_grid_movement(self, empty_game, direction):
        """Test movement on completely empty grid."""
        # Grid is already empty from fixture
        original_grid = empty_game.grid.copy()
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        # No movement should be possible
        assert result == False
        # Grid should remain unchanged
        np.testing.assert_array_equal(empty_game.grid, original_grid)
    
    # Test single tile movements
    @pytest.mark.parametrize("direction,tile_pos,expected_pos", [
        ("left", (2, 3), (2, 0)),    # Tile at (2,3) moves to (2,0)
        ("right", (2, 0), (2, 3)),   # Tile at (2,0) moves to (2,3)
        ("up", (3, 2), (0, 2)),      # Tile at (3,2) moves to (0,2)
        ("down", (0, 2), (3, 2)),    # Tile at (0,2) moves to (3,2)
    ])
    def test_single_tile_movement(self, empty_game, direction, tile_pos, expected_pos):
        """Test movement of a single tile in each direction."""
        empty_game.grid[tile_pos] = 2
        
        move_method = getattr(empty_game, f'move_{direction}')
        result = move_method()
        
        assert result == True
        assert empty_game.grid[expected_pos] == 2
        assert empty_game.grid[tile_pos] == 0
        assert np.sum(empty_game.grid > 0) == 1