#!/usr/bin/env python3

import os

from .core import Game2048


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_user_input() -> str:
    print("Enter move: [W]up [A]left [S]down [D]right [Q]uit")
    while True:
        try:
            key = input("> ").strip().lower()
            if key in ["w", "a", "s", "d", "q", "up", "left", "down", "right", "quit"]:
                # Convert word commands to letters
                key_map = {
                    "up": "w",
                    "left": "a",
                    "down": "s",
                    "right": "d",
                    "quit": "q",
                }
                return key_map.get(key, key)
            else:
                print(
                    "Invalid! Use W/A/S/D, arrow words (up/down/left/right), or Q to quit"
                )
        except (EOFError, KeyboardInterrupt):
            return "q"


def main():
    print("Welcome to 2048!")
    print("Combine tiles with the same number to reach 2048!")
    print()

    game = Game2048()

    while True:
        clear_screen()
        print("2048 Game")
        print("=" * 30)
        print(game)
        print(f"Game State: {game.get_state()}")

        if game.get_state() == "won":
            print("🎉 Congratulations! You've reached 2048!")
            if input("Continue playing? (y/n): ").lower() != "y":
                break
        elif game.get_state() == "lost":
            print("💀 Game Over! No more moves available.")
            if input("Play again? (y/n): ").lower() == "y":
                game = Game2048()
                continue
            else:
                break

        user_input = get_user_input()

        if user_input == "q":
            print("Thanks for playing!")
            break

        direction_map = {"w": "up", "a": "left", "s": "down", "d": "right"}

        direction = direction_map[user_input]
        if not game.move(direction):
            print(f"Cannot move {direction}! Try a different direction.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
