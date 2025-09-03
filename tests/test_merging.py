import numpy as np
import pytest

from game_2048.core import Game2048


class TestMerging:
    """Comprehensive tests for tile merging logic in the 2048 game, focusing on the _slide_line method."""

    @pytest.fixture
    def game(self):
        """Create a fresh game instance for testing."""
        return Game2048()

    # Basic tile merging tests
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # Basic merges
            ([2, 2, 0, 0], [4, 0, 0, 0], 4),
            ([4, 4, 0, 0], [8, 0, 0, 0], 8),
            ([8, 8, 0, 0], [16, 0, 0, 0], 16),
            ([16, 16, 0, 0], [32, 0, 0, 0], 32),
            ([32, 32, 0, 0], [64, 0, 0, 0], 64),
            ([64, 64, 0, 0], [128, 0, 0, 0], 128),
            ([128, 128, 0, 0], [256, 0, 0, 0], 256),
            ([256, 256, 0, 0], [512, 0, 0, 0], 512),
            ([512, 512, 0, 0], [1024, 0, 0, 0], 1024),
            ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),  # Maximum merge
            # Merges with gaps
            ([2, 0, 2, 0], [4, 0, 0, 0], 4),
            ([4, 0, 0, 4], [8, 0, 0, 0], 8),
            ([0, 8, 0, 8], [16, 0, 0, 0], 16),
            ([16, 0, 16, 0], [32, 0, 0, 0], 32),
            # Adjacent tiles that should merge
            ([2, 2, 4, 4], [4, 8, 0, 0], 12),  # 4 + 8 = 12 score
            ([8, 8, 2, 2], [16, 4, 0, 0], 20),  # 16 + 4 = 20 score
        ],
    )
    def test_basic_merging(self, game, line_input, expected_output, expected_score):
        """Test basic tile merging scenarios."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Chain prevention tests - ensure no triple merging
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # Classic chain prevention: [2,2,2,2] → [4,4,0,0] not [8,0,0,0]
            ([2, 2, 2, 2], [4, 4, 0, 0], 8),  # Two merges of 2+2=4 each
            ([4, 4, 4, 4], [8, 8, 0, 0], 16),  # Two merges of 4+4=8 each
            ([8, 8, 8, 8], [16, 16, 0, 0], 32),  # Two merges of 8+8=16 each
            # Three tiles: [4,4,4,0] → [8,4,0,0]
            ([4, 4, 4, 0], [8, 4, 0, 0], 8),
            ([2, 2, 2, 0], [4, 2, 0, 0], 4),
            ([8, 8, 8, 0], [16, 8, 0, 0], 16),
            # With gaps: [2,0,2,2] → [4,2,0,0]
            ([2, 0, 2, 2], [4, 2, 0, 0], 4),
            ([4, 0, 4, 4], [8, 4, 0, 0], 8),
            # Mixed with other tiles: [2,2,2,4] → [4,2,4,0]
            ([2, 2, 2, 4], [4, 2, 4, 0], 4),
            ([4, 4, 4, 2], [8, 4, 2, 0], 8),
        ],
    )
    def test_chain_prevention(self, game, line_input, expected_output, expected_score):
        """Test that chain reactions are prevented - no triple merging."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Mixed values that shouldn't merge
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # Different adjacent values - no merging
            ([2, 4, 8, 16], [2, 4, 8, 16], 0),
            ([4, 2, 8, 16], [4, 2, 8, 16], 0),
            ([2, 8, 4, 16], [2, 8, 4, 16], 0),
            # With gaps - should slide but not merge
            ([2, 0, 4, 0], [2, 4, 0, 0], 0),
            ([0, 4, 0, 8], [4, 8, 0, 0], 0),
            ([2, 0, 0, 4], [2, 4, 0, 0], 0),
            # Mixed scenario with some merges
            ([2, 2, 4, 8], [4, 4, 8, 0], 4),  # Only first pair merges
            ([2, 4, 4, 8], [2, 8, 8, 0], 8),  # Only middle pair merges
            ([2, 4, 8, 8], [2, 4, 16, 0], 16),  # Only last pair merges
        ],
    )
    def test_mixed_values_no_merge(
        self, game, line_input, expected_output, expected_score
    ):
        """Test that different values don't merge and slide correctly."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Large number merging
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # High value merges
            ([512, 512, 0, 0], [1024, 0, 0, 0], 1024),
            ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),  # Winning merge!
            ([256, 256, 512, 512], [512, 1024, 0, 0], 1536),  # 512 + 1024 = 1536
            # Large numbers with gaps
            ([512, 0, 512, 0], [1024, 0, 0, 0], 1024),
            ([1024, 0, 0, 1024], [2048, 0, 0, 0], 2048),
            ([256, 0, 256, 512], [512, 512, 0, 0], 512),
        ],
    )
    def test_large_number_merging(
        self, game, line_input, expected_output, expected_score
    ):
        """Test merging of large numbers including the winning 1024+1024=2048."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Score calculation tests
    @pytest.mark.parametrize(
        "line_input,expected_score",
        [
            # Single merge scores
            ([2, 2, 0, 0], 4),
            ([4, 4, 0, 0], 8),
            ([16, 16, 0, 0], 32),
            ([128, 128, 0, 0], 256),
            ([512, 512, 0, 0], 1024),
            # Multiple merge scores
            ([2, 2, 4, 4], 12),  # 4 + 8
            ([8, 8, 16, 16], 48),  # 16 + 32
            ([4, 4, 8, 8], 24),  # 8 + 16
            ([2, 2, 2, 2], 8),  # 4 + 4 (two separate merges)
            # Complex scoring
            ([4, 4, 4, 4], 16),  # 8 + 8
            ([8, 8, 8, 8], 32),  # 16 + 16
            ([2, 2, 8, 8], 20),  # 4 + 16
        ],
    )
    def test_score_calculation(self, game, line_input, expected_score):
        """Test that score is calculated correctly for various merge scenarios."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        assert score == expected_score

    # Multiple merges in one line
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # Two separate merges
            ([2, 2, 4, 4], [4, 8, 0, 0], 12),
            ([4, 4, 8, 8], [8, 16, 0, 0], 24),
            ([8, 8, 16, 16], [16, 32, 0, 0], 48),
            # With gaps between pairs - only first 4 elements used
            ([2, 2, 0, 4], [4, 4, 0, 0], 4),  # 2+2=4, single 4 remains
            # Different sized merges
            ([2, 2, 8, 8], [4, 16, 0, 0], 20),  # 4 + 16
            ([4, 4, 2, 2], [8, 4, 0, 0], 12),  # 8 + 4
            ([16, 16, 4, 4], [32, 8, 0, 0], 40),  # 32 + 8
        ],
    )
    def test_multiple_merges_in_line(
        self, game, line_input, expected_output, expected_score
    ):
        """Test multiple merges occurring in a single line."""
        line = np.array(
            line_input[:4], dtype=int
        )  # Ensure we only take first 4 elements
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Merging with gaps between tiles
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # Single gap scenarios
            ([2, 0, 2, 0], [4, 0, 0, 0], 4),
            ([0, 4, 0, 4], [8, 0, 0, 0], 8),
            ([8, 0, 0, 8], [16, 0, 0, 0], 16),
            # Multiple gaps
            ([2, 0, 0, 2], [4, 0, 0, 0], 4),
            ([0, 0, 4, 4], [8, 0, 0, 0], 8),
            ([8, 0, 8, 0], [16, 0, 0, 0], 16),
            # Complex gaps with multiple merges
            ([2, 0, 2, 4], [4, 4, 0, 0], 4),  # 2+2=4, single 4 remains
            ([4, 0, 4, 0], [8, 0, 0, 0], 8),  # 4+4=8
            # No merge due to gaps and different values
            ([2, 0, 4, 0], [2, 4, 0, 0], 0),
            ([0, 8, 0, 16], [8, 16, 0, 0], 0),
        ],
    )
    def test_merging_with_gaps(self, game, line_input, expected_output, expected_score):
        """Test that tiles merge correctly across gaps."""
        line = np.array(
            line_input[:4], dtype=int
        )  # Ensure we only take first 4 elements
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Edge case: maximum tiles merging
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # The ultimate merge - reaching 2048
            ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),
            ([0, 1024, 1024, 0], [2048, 0, 0, 0], 2048),
            ([1024, 0, 1024, 0], [2048, 0, 0, 0], 2048),
            ([0, 0, 1024, 1024], [2048, 0, 0, 0], 2048),
            # Beyond 2048 (if game continues)
            ([2048, 2048, 0, 0], [4096, 0, 0, 0], 4096),
            ([4096, 4096, 0, 0], [8192, 0, 0, 0], 8192),
            # Mixed high values
            ([512, 512, 1024, 1024], [1024, 2048, 0, 0], 3072),  # 1024 + 2048
        ],
    )
    def test_maximum_tile_merging(
        self, game, line_input, expected_output, expected_score
    ):
        """Test merging of maximum value tiles (1024+1024=2048 and beyond)."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Empty line handling
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # Completely empty
            ([0, 0, 0, 0], [0, 0, 0, 0], 0),
            # Single tiles (no merging possible)
            ([2, 0, 0, 0], [2, 0, 0, 0], 0),
            ([0, 4, 0, 0], [4, 0, 0, 0], 0),
            ([0, 0, 8, 0], [8, 0, 0, 0], 0),
            ([0, 0, 0, 16], [16, 0, 0, 0], 0),
            # Mostly empty with non-matching tiles
            ([2, 0, 4, 0], [2, 4, 0, 0], 0),
            ([0, 8, 0, 16], [8, 16, 0, 0], 0),
        ],
    )
    def test_empty_line_handling(
        self, game, line_input, expected_output, expected_score
    ):
        """Test handling of empty lines and single tiles."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score

    # Test that no triple merging occurs in various scenarios
    @pytest.mark.parametrize(
        "line_input,expected_output",
        [
            # Sequential identical tiles
            ([2, 2, 2, 0], [4, 2, 0, 0]),  # First two merge, third remains
            ([4, 4, 4, 0], [8, 4, 0, 0]),
            ([8, 8, 8, 0], [16, 8, 0, 0]),
            # Four identical tiles - should create two pairs
            ([2, 2, 2, 2], [4, 4, 0, 0]),
            ([4, 4, 4, 4], [8, 8, 0, 0]),
            # With gaps but still three identical
            (
                [2, 0, 2, 2],
                [4, 2, 0, 0],
            ),  # First and third merge to 4, fourth becomes 2
            ([4, 0, 4, 4], [8, 4, 0, 0]),
            # Five identical values (taking first 4)
            ([2, 2, 2, 2, 2], [4, 4, 0, 0]),  # Should process as [2,2,2,2]
        ],
    )
    def test_no_triple_merging_prevention(self, game, line_input, expected_output):
        """Test that triple merging is prevented in all scenarios."""
        line = np.array(
            line_input[:4], dtype=int
        )  # Ensure we only take first 4 elements
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))

    # Integration test: test that _slide_line affects the game's moved flag
    def test_moved_flag_integration(self, game):
        """Test that _slide_line properly sets the moved flag when tiles merge."""
        # Reset the moved flag
        game.moved = False

        # Test line with merging
        line_with_merge = np.array([2, 2, 0, 0], dtype=int)
        result, score = game._slide_line(line_with_merge)

        # moved flag should be set to True because tiles merged
        assert game.moved == True

        # Reset and test line without merging but with sliding
        game.moved = False
        line_with_slide = np.array([2, 0, 4, 0], dtype=int)
        result, score = game._slide_line(line_with_slide)

        # moved flag should remain False because _slide_line only sets it for merges, not slides
        # The sliding detection happens at the move_left level
        assert game.moved == False

    # Comprehensive edge cases
    @pytest.mark.parametrize(
        "line_input,expected_output,expected_score",
        [
            # All zeros
            ([0, 0, 0, 0], [0, 0, 0, 0], 0),
            # Single non-zero tile in different positions
            ([2, 0, 0, 0], [2, 0, 0, 0], 0),
            ([0, 2, 0, 0], [2, 0, 0, 0], 0),
            ([0, 0, 2, 0], [2, 0, 0, 0], 0),
            ([0, 0, 0, 2], [2, 0, 0, 0], 0),
            # Two identical tiles in different positions
            ([2, 2, 0, 0], [4, 0, 0, 0], 4),
            ([2, 0, 2, 0], [4, 0, 0, 0], 4),
            ([2, 0, 0, 2], [4, 0, 0, 0], 4),
            ([0, 2, 2, 0], [4, 0, 0, 0], 4),
            ([0, 2, 0, 2], [4, 0, 0, 0], 4),
            ([0, 0, 2, 2], [4, 0, 0, 0], 4),
            # Maximum complexity: alternating pattern
            ([2, 4, 2, 4], [2, 4, 2, 4], 0),  # No adjacent identical tiles
            ([2, 2, 4, 4], [4, 8, 0, 0], 12),  # Two pairs merge
            ([4, 2, 4, 2], [4, 2, 4, 2], 0),  # No adjacent identical tiles
        ],
    )
    def test_comprehensive_edge_cases(
        self, game, line_input, expected_output, expected_score
    ):
        """Test comprehensive edge cases for the _slide_line method."""
        line = np.array(line_input, dtype=int)
        result, score = game._slide_line(line)

        np.testing.assert_array_equal(result, np.array(expected_output, dtype=int))
        assert score == expected_score
