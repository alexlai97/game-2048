#!/usr/bin/env python3

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QPalette, QColor
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
            0: ("#cdc1b4", "#776e65"),     # Empty
            2: ("#eee4da", "#776e65"),     # 2
            4: ("#ede0c8", "#776e65"),     # 4
            8: ("#f2b179", "#f9f6f2"),     # 8
            16: ("#f59563", "#f9f6f2"),    # 16
            32: ("#f67c5f", "#f9f6f2"),    # 32
            64: ("#f65e3b", "#f9f6f2"),    # 64
            128: ("#edcf72", "#f9f6f2"),   # 128
            256: ("#edcc61", "#f9f6f2"),   # 256
            512: ("#edc850", "#f9f6f2"),   # 512
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
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        self.setWindowTitle("2048 - PySide6")
        self.setFixedSize(400, 500)
        
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
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title_label = QLabel("2048")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 32, QFont.Bold))
        title_label.setStyleSheet("color: #776e65; margin-bottom: 10px;")
        main_layout.addWidget(title_label)
        
        # Score and controls layout
        header_layout = QHBoxLayout()
        
        # Score
        self.score_label = QLabel(f"Score: {self.game.score}")
        self.score_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(self.score_label)
        
        header_layout.addStretch()
        
        # New Game button
        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self.restart_game)
        header_layout.addWidget(new_game_btn)
        
        main_layout.addLayout(header_layout)
        
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
            
        main_layout.addWidget(board_frame)
        
        # Instructions
        instructions = QLabel(
            "Use arrow keys or WASD to move tiles\n"
            "Press R to restart • Press Q to quit"
        )
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setFont(QFont("Arial", 12))
        instructions.setStyleSheet("color: #776e65; margin-top: 10px;")
        main_layout.addWidget(instructions)
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        # Arrow keys and WASD
        shortcuts = {
            Qt.Key_Up: 'up',
            Qt.Key_Down: 'down',
            Qt.Key_Left: 'left',
            Qt.Key_Right: 'right',
            Qt.Key_W: 'up',
            Qt.Key_S: 'down',
            Qt.Key_A: 'left',
            Qt.Key_D: 'right',
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
                
    def make_move(self, direction):
        if self.game.move(direction):
            self.update_display()
            # Small delay to show the move, then check game state
            QTimer.singleShot(100, self.check_game_state)
            
    def check_game_state(self):
        state = self.game.get_state()
        if state == "won":
            reply = QMessageBox.question(
                self, 
                "You Won!",
                f"Congratulations! You reached 2048!\n"
                f"Score: {self.game.score}\n\n"
                f"Continue playing?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.No:
                self.restart_game()
        elif state == "lost":
            reply = QMessageBox.question(
                self, 
                "Game Over",
                f"No more moves available!\n"
                f"Final Score: {self.game.score}\n\n"
                f"Play again?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.restart_game()
            else:
                self.close()
                
    def restart_game(self):
        self.game = Game2048()
        self.update_display()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("2048")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')  # Modern, cross-platform style
    
    window = Game2048PySide6()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()