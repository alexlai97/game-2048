#!/usr/bin/env python3

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from ..core import Game2048
from .base import BaseAgent


@dataclass
class GameResult:
    """Container for individual game results."""

    score: int
    moves: int
    max_tile: int
    won: bool
    duration_seconds: float
    final_state: str


class AIRunner:
    """
    Runner class for testing AI agents on 2048.

    Provides functionality to run single games, batch tests,
    and collect performance statistics.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the AI runner.

        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose

    def run_single_game(
        self,
        agent: BaseAgent,
        max_moves: int = 10000,
        game_callback: Optional[Callable] = None,
    ) -> GameResult:
        """
        Run a single game with the specified agent.

        Args:
            agent: AI agent to play the game
            max_moves: Maximum number of moves before terminating
            game_callback: Optional callback function called after each move
                          with signature: callback(game, move, move_count)

        Returns:
            GameResult: Results from the completed game
        """
        game = Game2048()
        agent.reset()

        move_count = 0
        start_time = time.time()

        if self.verbose:
            print(f"\nStarting game with {agent.get_name()}...")

        while game.get_state() == "ongoing" and move_count < max_moves:
            try:
                move = agent.get_move(game)
                if not game.move(move):
                    if self.verbose:
                        print(f"Warning: Invalid move '{move}' attempted")
                    break

                move_count += 1

                # Call callback if provided
                if game_callback:
                    game_callback(game, move, move_count)

            except ValueError as e:
                if self.verbose:
                    print(f"Agent error: {e}")
                break
            except Exception as e:
                if self.verbose:
                    print(f"Unexpected error: {e}")
                break

        end_time = time.time()
        duration = end_time - start_time

        # Determine game outcome
        final_state = game.get_state()
        won = final_state == "won"
        max_tile = int(game.grid.max())

        # Update agent statistics
        agent.game_finished(game.score, won, max_tile)

        result = GameResult(
            score=game.score,
            moves=move_count,
            max_tile=max_tile,
            won=won,
            duration_seconds=duration,
            final_state=final_state,
        )

        if self.verbose:
            print(f"Game finished: {final_state}")
            print(
                f"Score: {result.score}, Moves: {result.moves}, Max tile: {result.max_tile}"
            )
            print(f"Duration: {duration:.2f}s")

        return result

    def run_batch(
        self, agent: BaseAgent, num_games: int, max_moves: int = 10000
    ) -> list[GameResult]:
        """
        Run multiple games with the agent and collect results.

        Args:
            agent: AI agent to test
            num_games: Number of games to run
            max_moves: Maximum moves per game

        Returns:
            List[GameResult]: Results from all games
        """
        results = []
        start_time = time.time()

        if self.verbose:
            print(f"\nRunning {num_games} games with {agent.get_name()}...")
            print("=" * 50)

        for i in range(num_games):
            if self.verbose and (i + 1) % max(1, num_games // 10) == 0:
                print(f"Progress: {i + 1}/{num_games} games completed")

            result = self.run_single_game(agent, max_moves, game_callback=None)
            results.append(result)

        total_time = time.time() - start_time

        if self.verbose:
            print(f"\nBatch completed in {total_time:.2f}s")
            self.print_batch_statistics(results, agent)

        return results

    def print_batch_statistics(self, results: list[GameResult], agent: BaseAgent):
        """Print comprehensive statistics for a batch of games."""
        if not results:
            print("No results to analyze")
            return

        # Basic statistics
        total_games = len(results)
        wins = sum(1 for r in results if r.won)
        win_rate = wins / total_games

        scores = [r.score for r in results]
        moves = [r.moves for r in results]
        max_tiles = [r.max_tile for r in results]
        durations = [r.duration_seconds for r in results]

        # Score statistics
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)

        # Move statistics
        avg_moves = sum(moves) / len(moves)
        max_moves = max(moves)
        min_moves = min(moves)

        # Tile statistics
        tile_counts = {}
        for tile in max_tiles:
            tile_counts[tile] = tile_counts.get(tile, 0) + 1

        print("\n" + "=" * 60)
        print(f"BATCH STATISTICS FOR {agent.get_name().upper()}")
        print("=" * 60)
        print(f"Games played:     {total_games}")
        print(f"Wins:            {wins} ({win_rate:.1%})")
        print(f"Losses:          {total_games - wins}")
        print()
        print("SCORE STATISTICS:")
        print(f"  Average:       {avg_score:.1f}")
        print(f"  Maximum:       {max_score}")
        print(f"  Minimum:       {min_score}")
        print()
        print("MOVE STATISTICS:")
        print(f"  Average:       {avg_moves:.1f}")
        print(f"  Maximum:       {max_moves}")
        print(f"  Minimum:       {min_moves}")
        print()
        print("MAX TILE DISTRIBUTION:")
        for tile in sorted(tile_counts.keys(), reverse=True):
            count = tile_counts[tile]
            percentage = count / total_games * 100
            print(f"  {tile:4d}:         {count:3d} games ({percentage:5.1f}%)")
        print()
        print(f"Average game time: {sum(durations) / len(durations):.2f}s")
        print("=" * 60)

    def save_results(
        self,
        results: list[GameResult],
        agent: BaseAgent,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save batch results to JSON file.

        Args:
            results: Game results to save
            agent: Agent that played the games
            filename: Optional custom filename

        Returns:
            str: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agent.get_name()}_results_{timestamp}.json"

        # Convert results to serializable format
        data = {
            "agent_name": agent.get_name(),
            "agent_class": agent.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "total_games": len(results),
            "agent_statistics": agent.get_statistics(),
            "games": [
                {
                    "score": int(r.score),
                    "moves": int(r.moves),
                    "max_tile": int(r.max_tile),
                    "won": bool(r.won),
                    "duration_seconds": float(r.duration_seconds),
                    "final_state": str(r.final_state),
                }
                for r in results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"Results saved to: {filename}")

        return filename


def main():
    """Main function for running AI tests from command line."""
    if len(sys.argv) < 3:
        print("Usage: 2048-ai <agent_class> <num_games>")
        print("Example: 2048-ai RandomAgent 100")
        print("Alternative: python -m game_2048.agents.runner RandomAgent 100")
        sys.exit(1)

    agent_name = sys.argv[1]
    num_games = int(sys.argv[2])

    # Import and create agent
    if agent_name == "RandomAgent":
        from .random import RandomAgent

        agent = RandomAgent()
    elif agent_name == "GreedyAgent":
        from .greedy import GreedyAgent

        agent = GreedyAgent()
    elif agent_name == "MCTSAgent":
        from .mcts import MCTSAgent

        agent = MCTSAgent()
    elif agent_name == "MinimaxAgent":
        from .minimax import MinimaxAgent

        agent = MinimaxAgent()
    elif agent_name == "ExpectimaxAgent":
        from .expectimax import ExpectimaxAgent

        agent = ExpectimaxAgent()
    else:
        print(f"Unknown agent: {agent_name}")
        print(
            "Available agents: RandomAgent, GreedyAgent, MCTSAgent, MinimaxAgent, ExpectimaxAgent"
        )
        sys.exit(1)

    # Run batch test
    runner = AIRunner(verbose=True)
    results = runner.run_batch(agent, num_games)

    # Save results
    runner.save_results(results, agent)


if __name__ == "__main__":
    main()
