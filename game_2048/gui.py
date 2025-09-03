#!/usr/bin/env python3

import os
import sys

from PySide6.QtCore import (
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .agents.expectimax import ExpectimaxAgent
from .agents.greedy import GreedyAgent
from .agents.mcts import MCTSAgent
from .agents.minimax import MinimaxAgent
from .agents.random import RandomAgent
from .core import Game2048


class TileWidget(QLabel):
    def __init__(self, value=0):
        super().__init__()
        self.value = value
        self.setFixedSize(80, 80)
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Arial", 16, QFont.Bold))
        self.update_appearance()

    def set_value(self, value):
        self.value = value
        self.update_appearance()

    def update_appearance(self):
        # Color scheme based on tile values
        colors = {
            0: ("#cdc1b4", "#776e65"),  # Empty
            2: ("#eee4da", "#776e65"),  # 2
            4: ("#ede0c8", "#776e65"),  # 4
            8: ("#f2b179", "#f9f6f2"),  # 8
            16: ("#f59563", "#f9f6f2"),  # 16
            32: ("#f67c5f", "#f9f6f2"),  # 32
            64: ("#f65e3b", "#f9f6f2"),  # 64
            128: ("#edcf72", "#f9f6f2"),  # 128
            256: ("#edcc61", "#f9f6f2"),  # 256
            512: ("#edc850", "#f9f6f2"),  # 512
            1024: ("#edc53f", "#f9f6f2"),  # 1024
            2048: ("#edc22e", "#f9f6f2"),  # 2048
            4096: ("#3c3a32", "#f9f6f2"),  # 4096+
        }

        bg_color, text_color = colors.get(self.value, colors[4096])

        if self.value == 0:
            self.setText("")
        else:
            self.setText(str(self.value))

        # Adjust font size for larger numbers
        font_size = 16
        if self.value >= 1000:
            font_size = 12
        elif self.value >= 100:
            font_size = 14

        font = QFont("Arial", font_size, QFont.Bold)
        self.setFont(font)

        # Set style with rounded corners and shadows
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                border-radius: 8px;
                border: 2px solid #bbada0;
            }}
        """)


class Game2048PySide6(QMainWindow):
    def __init__(self):
        super().__init__()
        self.game = Game2048()
        self.tiles = []

        # AI-related attributes
        self.ai_mode = False
        self.ai_agent = None
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.ai_make_move)
        self.ai_speed = 5  # moves per second
        self.ai_running = False
        self.move_count = 0
        self.last_move = None

        # Sound system
        self.sounds_enabled = True
        self.sound_effects = {}
        self.init_sounds()

        # Visualization attributes
        self.move_highlight_timer = QTimer()
        self.move_highlight_timer.timeout.connect(self.clear_move_highlight)
        self.move_highlight_timer.setSingleShot(True)

        # Game over dialog guard
        self.game_over_shown = False

        self.setup_ui()
        self.update_display()

    def init_sounds(self):
        """Initialize sound effects for the game."""
        sounds_dir = os.path.join(os.path.dirname(__file__), "sounds")

        sound_files = {
            "move": "move.wav",
            "merge": "merge.wav",
            "win": "win.wav",
            "lose": "lose.wav",
            "spawn": "spawn.wav",
        }

        for sound_name, filename in sound_files.items():
            sound_path = os.path.join(sounds_dir, filename)
            if os.path.exists(sound_path):
                effect = QSoundEffect()
                effect.setSource(QUrl.fromLocalFile(sound_path))
                effect.setVolume(0.5)  # 50% volume
                self.sound_effects[sound_name] = effect

    def play_sound(self, sound_name):
        """Play a sound effect if sounds are enabled."""
        if self.sounds_enabled and sound_name in self.sound_effects:
            self.sound_effects[sound_name].play()

    def toggle_sounds(self, checked):
        """Toggle sound effects on/off."""
        self.sounds_enabled = checked
        self.update_sound_button()

    def update_sound_button(self):
        """Update the sound button appearance based on current state."""
        if self.sounds_enabled:
            self.sound_button.setText("🔊 Sound")
            # Use normal button colors when sounds are on
            self.sound_button.setStyleSheet("""
                QPushButton {
                    background-color: #8f7a66;
                    color: #f9f6f2;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #9f8a76;
                }
                QPushButton:pressed {
                    background-color: #7f6a56;
                }
            """)
        else:
            self.sound_button.setText("🔇 Muted")
            # Use a grayed-out appearance when sounds are off
            self.sound_button.setStyleSheet("""
                QPushButton {
                    background-color: #c4b7a6;
                    color: #8f7a66;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #d4c7b6;
                }
                QPushButton:pressed {
                    background-color: #b4a796;
                }
            """)

    def setup_ui(self):
        self.setWindowTitle("2048 - PySide6 with AI")
        self.setFixedSize(650, 600)

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #faf8ef;
            }
            QLabel {
                color: #776e65;
            }
            QPushButton {
                background-color: #8f7a66;
                color: #f9f6f2;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9f8a76;
            }
            QPushButton:pressed {
                background-color: #7f6a56;
            }
            QComboBox {
                background-color: #8f7a66;
                color: #f9f6f2;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                background-color: #bbada0;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background-color: #8f7a66;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bbada0;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout - horizontal to fit controls beside game
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left side - game area
        game_layout = QVBoxLayout()
        game_layout.setSpacing(20)

        # Title
        title_label = QLabel("2048")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 32, QFont.Bold))
        title_label.setStyleSheet("color: #776e65; margin-bottom: 10px;")
        game_layout.addWidget(title_label)

        # Score and controls layout
        header_layout = QHBoxLayout()

        # Score
        self.score_label = QLabel(f"Score: {self.game.score}")
        self.score_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(self.score_label)

        header_layout.addStretch()

        # Sound toggle
        self.sound_button = QPushButton("🔊 Sound")
        self.sound_button.setCheckable(True)
        self.sound_button.setChecked(True)
        self.sound_button.clicked.connect(self.toggle_sounds)
        self.update_sound_button()  # Initialize button appearance
        header_layout.addWidget(self.sound_button)

        # New Game button
        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self.restart_game)
        header_layout.addWidget(new_game_btn)

        game_layout.addLayout(header_layout)

        # Game board frame
        board_frame = QFrame()
        board_frame.setStyleSheet("""
            QFrame {
                background-color: #bbada0;
                border-radius: 8px;
                padding: 8px;
            }
        """)

        # Game grid
        self.grid_layout = QGridLayout(board_frame)
        self.grid_layout.setSpacing(8)

        # Create tile widgets
        for i in range(4):
            row = []
            for j in range(4):
                tile = TileWidget()
                self.grid_layout.addWidget(tile, i, j)
                row.append(tile)
            self.tiles.append(row)

        game_layout.addWidget(board_frame)

        # Instructions
        instructions = QLabel(
            "Use arrow keys or WASD to move tiles\nPress R to restart • Press Q to quit"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setFont(QFont("Arial", 12))
        instructions.setStyleSheet("color: #776e65; margin-top: 10px;")
        game_layout.addWidget(instructions)

        # Add game layout to main layout
        main_layout.addLayout(game_layout)

        # Right side - AI control panel
        self.setup_ai_panel(main_layout)

        # Setup keyboard shortcuts
        self.setup_shortcuts()

    def setup_ai_panel(self, main_layout):
        """Set up the AI control panel on the right side."""
        ai_panel = QGroupBox("AI Controls")
        ai_panel.setFixedWidth(250)
        ai_layout = QVBoxLayout(ai_panel)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Human", "AI"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        ai_layout.addLayout(mode_layout)

        # AI agent selector
        agent_layout = QHBoxLayout()
        agent_layout.addWidget(QLabel("Agent:"))
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(
            [
                "RandomAgent",
                "GreedyAgent",
                "MinimaxAgent",
                "ExpectimaxAgent",
                "MCTSAgent",
            ]
        )
        self.agent_combo.setEnabled(False)
        self.agent_combo.currentTextChanged.connect(self.on_agent_changed)
        agent_layout.addWidget(self.agent_combo)
        ai_layout.addLayout(agent_layout)

        # Speed control
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speed (moves/sec):"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(5)
        self.speed_slider.setEnabled(False)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("5 moves/sec")
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        ai_layout.addLayout(speed_layout)

        # AI control buttons
        button_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_ai_play)
        button_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_ai)
        button_layout.addWidget(self.stop_btn)
        ai_layout.addLayout(button_layout)

        # Statistics panel
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.moves_label = QLabel("Moves: 0")
        self.last_move_label = QLabel("Last move: -")
        self.agent_status_label = QLabel("Status: Idle")

        stats_layout.addWidget(self.moves_label)
        stats_layout.addWidget(self.last_move_label)
        stats_layout.addWidget(self.agent_status_label)

        ai_layout.addWidget(stats_group)
        ai_layout.addStretch()

        main_layout.addWidget(ai_panel)

    def setup_shortcuts(self):
        # Arrow keys and WASD
        shortcuts = {
            Qt.Key_Up: "up",
            Qt.Key_Down: "down",
            Qt.Key_Left: "left",
            Qt.Key_Right: "right",
            Qt.Key_W: "up",
            Qt.Key_S: "down",
            Qt.Key_A: "left",
            Qt.Key_D: "right",
        }

        for key, direction in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda d=direction: self.make_move(d))

        # Restart and quit
        QShortcut(QKeySequence(Qt.Key_R), self).activated.connect(self.restart_game)
        QShortcut(QKeySequence(Qt.Key_Q), self).activated.connect(self.close)

    def update_display(self):
        # Update score
        self.score_label.setText(f"Score: {self.game.score}")

        # Update tiles
        for i in range(4):
            for j in range(4):
                value = self.game.grid[i, j]
                self.tiles[i][j].set_value(value)

    def highlight_move_direction(self, direction):
        """Visually highlight the move direction for 500ms."""
        # Create a temporary overlay to show move direction
        self.show_move_indicator(direction)

        # Clear highlight after 500ms
        self.move_highlight_timer.start(500)

    def show_move_indicator(self, direction):
        """Show a visual indicator of the move direction."""
        # Add visual feedback by temporarily modifying the board frame
        board_frame = None
        for widget in self.findChildren(QFrame):
            if widget.styleSheet().find("background-color: #bbada0") != -1:
                board_frame = widget
                break

        if board_frame:
            direction_colors = {
                "up": "#4a90e2",  # Blue
                "down": "#e24a4a",  # Red
                "left": "#4ae24a",  # Green
                "right": "#e2a44a",  # Orange
            }

            color = direction_colors.get(direction, "#8f7a66")

            # Temporarily change border color to indicate direction
            original_style = board_frame.styleSheet()
            highlighted_style = original_style.replace(
                "border-radius: 8px;", f"border-radius: 8px; border: 4px solid {color};"
            )
            board_frame.setStyleSheet(highlighted_style)

            # Store original style for restoration
            if not hasattr(self, "original_board_style"):
                self.original_board_style = original_style

    def clear_move_highlight(self):
        """Clear the move direction highlight."""
        # Restore original board frame style
        if hasattr(self, "original_board_style"):
            for widget in self.findChildren(QFrame):
                if "border: 4px solid" in widget.styleSheet():
                    widget.setStyleSheet(self.original_board_style)
                    break

    def on_mode_changed(self, mode):
        """Handle mode change between Human and AI."""
        self.ai_mode = mode == "AI"

        # Enable/disable AI controls
        self.agent_combo.setEnabled(self.ai_mode)
        self.speed_slider.setEnabled(self.ai_mode)
        self.play_btn.setEnabled(self.ai_mode)

        if self.ai_mode:
            self.on_agent_changed(self.agent_combo.currentText())
            self.agent_status_label.setText("Status: Ready")
        else:
            self.stop_ai()
            self.agent_status_label.setText("Status: Human Mode")

    def on_agent_changed(self, agent_name):
        """Handle AI agent selection."""
        if not self.ai_mode:
            return

        self.stop_ai()  # Stop current agent if running

        # Create new agent instance
        if agent_name == "RandomAgent":
            self.ai_agent = RandomAgent()
        elif agent_name == "GreedyAgent":
            self.ai_agent = GreedyAgent()
        elif agent_name == "MinimaxAgent":
            self.ai_agent = MinimaxAgent()
        elif agent_name == "ExpectimaxAgent":
            self.ai_agent = ExpectimaxAgent()
        elif agent_name == "MCTSAgent":
            self.ai_agent = MCTSAgent(simulations=10)  # Reduced for GUI responsiveness
        else:
            self.ai_agent = None

        self.agent_status_label.setText(f"Status: {agent_name} Ready")

    def on_speed_changed(self, value):
        """Handle speed slider change."""
        self.ai_speed = value
        self.speed_label.setText(f"{value} moves/sec")

        # Update timer interval if running
        if self.ai_running:
            self.ai_timer.setInterval(int(1000 / value))

    def toggle_ai_play(self):
        """Toggle AI play/pause."""
        if self.ai_running:
            self.pause_ai()
        else:
            self.start_ai()

    def start_ai(self):
        """Start AI execution."""
        if not self.ai_mode or not self.ai_agent:
            return

        # Reset game over flag when starting AI
        self.game_over_shown = False
        self.ai_running = True
        self.play_btn.setText("Pause")
        self.stop_btn.setEnabled(True)
        self.agent_status_label.setText(f"Status: {self.ai_agent.get_name()} Playing")

        # Set timer interval based on speed
        interval = int(1000 / self.ai_speed)  # Convert to milliseconds
        self.ai_timer.setInterval(interval)
        self.ai_timer.start()

    def pause_ai(self):
        """Pause AI execution."""
        self.ai_running = False
        self.ai_timer.stop()
        self.play_btn.setText("Play")
        self.agent_status_label.setText(f"Status: {self.ai_agent.get_name()} Paused")

    def stop_ai(self):
        """Stop AI execution completely."""
        self.ai_running = False
        self.ai_timer.stop()
        self.play_btn.setText("Play")
        self.stop_btn.setEnabled(False)

        if self.ai_agent:
            self.agent_status_label.setText(f"Status: {self.ai_agent.get_name()} Ready")
        else:
            self.agent_status_label.setText("Status: Ready")

    def ai_make_move(self):
        """Make one AI move."""
        if not self.ai_mode or not self.ai_agent or not self.ai_running:
            return

        try:
            # Check if game is over
            if self.game.get_state() != "ongoing":
                self.stop_ai()
                return

            # Get AI move
            move = self.ai_agent.get_move(self.game)
            self.last_move = move

            # Highlight the move direction
            self.highlight_move_direction(move)

            # Make the move
            if self.game.move(move):
                # Play appropriate sounds
                self.play_sound("move")
                if self.game.merged_tiles > 0:
                    self.play_sound("merge")

                self.move_count += 1
                self.update_display()
                self.update_ai_stats()

                # Check game state after a short delay
                QTimer.singleShot(50, self.check_ai_game_state)
            else:
                # Invalid move, stop AI
                self.agent_status_label.setText("Status: Invalid move, stopped")
                self.stop_ai()

        except Exception as e:
            self.agent_status_label.setText(f"Status: Error - {str(e)}")
            self.stop_ai()

    def update_ai_stats(self):
        """Update AI statistics display."""
        self.moves_label.setText(f"Moves: {self.move_count}")
        if self.last_move:
            self.last_move_label.setText(f"Last move: {self.last_move}")

    def check_ai_game_state(self):
        """Check game state during AI play."""
        state = self.game.get_state()

        # Only show game over dialog once
        if self.game_over_shown:
            return

        if state == "won":
            self.play_sound("win")
            self.stop_ai()
            self.game_over_shown = True
            self.agent_status_label.setText("Status: Game Won!")
            QMessageBox.information(
                self,
                "AI Won!",
                f"{self.ai_agent.get_name()} reached 2048!\n"
                f"Score: {self.game.score}\n"
                f"Moves: {self.move_count}",
            )
        elif state == "lost":
            self.play_sound("lose")
            self.stop_ai()
            self.game_over_shown = True
            self.agent_status_label.setText("Status: Game Over")
            QMessageBox.information(
                self,
                "Game Over",
                f"{self.ai_agent.get_name()} finished\n"
                f"Final Score: {self.game.score}\n"
                f"Moves: {self.move_count}",
            )

    def make_move(self, direction):
        # Only allow human moves in human mode
        if self.ai_mode:
            return

        if self.game.move(direction):
            # Play appropriate sounds
            self.play_sound("move")
            if self.game.merged_tiles > 0:
                self.play_sound("merge")
            if self.game.spawned_new_tile:
                # Optional: play spawn sound (might be too frequent)
                pass

            self.update_display()
            # Small delay to show the move, then check game state
            QTimer.singleShot(100, self.check_game_state)

    def check_game_state(self):
        state = self.game.get_state()
        if state == "won":
            self.play_sound("win")
            reply = QMessageBox.question(
                self,
                "You Won!",
                f"Congratulations! You reached 2048!\\n"
                f"Score: {self.game.score}\\n\\n"
                f"Continue playing?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.No:
                self.restart_game()
        elif state == "lost":
            self.play_sound("lose")
            reply = QMessageBox.question(
                self,
                "Game Over",
                f"No more moves available!\\n"
                f"Final Score: {self.game.score}\\n\\n"
                f"Play again?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.restart_game()
            else:
                self.close()

    def restart_game(self):
        # Stop AI if running
        if self.ai_running:
            self.stop_ai()

        # Reset game over flag for new game
        self.game_over_shown = False

        self.game = Game2048()
        self.move_count = 0
        self.last_move = None
        self.update_display()

        # Reset AI statistics
        if hasattr(self, "moves_label"):
            self.moves_label.setText("Moves: 0")
            self.last_move_label.setText("Last move: -")

        # Update agent status
        if self.ai_mode and self.ai_agent:
            self.agent_status_label.setText(f"Status: {self.ai_agent.get_name()} Ready")
        elif self.ai_mode:
            self.agent_status_label.setText("Status: Ready")
        else:
            self.agent_status_label.setText("Status: Human Mode")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("2048")
    app.setApplicationVersion("1.0")

    # Set application style
    app.setStyle("Fusion")  # Modern, cross-platform style

    window = Game2048PySide6()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
