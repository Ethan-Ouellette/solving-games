#!/usr/bin/env python3
"""
Tic Stac Toe 4×4×4 terminal game with optional Minimax AI and undo feature.
Features:
- Gravity on/off via --no-gravity flag
- Human vs. Human or Human vs. AI (--ai O or --ai X)
- Adjustable Minimax depth (--depth)
- Immediate win/block detection
- Clear, consistent display formatting
- Input validation and improved undo (handles AI)
- Undo available even after game end
"""
import argparse
import ast
import os
import sys
from typing import List, Tuple, Optional

import numpy as np
import math

# Constants
O, X = 0, 1
EMPTY = -1
BOARD_SIZE = 4
CELL_WIDTH = 5
WIN_LINES_FILE = "winning_lines_4x4x4.txt"

# Load winning lines
def load_win_lines(path: str) -> List[List[Tuple[int,int,int]]]:
    try:
        with open(path, "r") as f:
            return [ast.literal_eval(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"Error: win-lines file '{path}' not found.")
        sys.exit(1)

win_lines = load_win_lines(WIN_LINES_FILE)
space_to_win_lines = {pos: [] for pos in np.ndindex(*(BOARD_SIZE,)*3)}
for line in win_lines:
    for pos in line:
        space_to_win_lines[pos].append(line)

# ANSI helpers
def color_text(text: str, code: str = "91") -> str:
    return f"\033[{code}m{text}\033[0m"

# Board symbol formatting
def format_symbol(sym: str) -> str:
    return f"{sym:^{CELL_WIDTH}}"

# Type alias
Move = Tuple[int,int,int]

# Board class
class Board:
    def __init__(self):
        self.grid = np.full((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), EMPTY, dtype=np.int8)

    def add_piece(self, player: int, x: int, y: int, z: int) -> None:
        self.grid[x, y, z] = player

    def remove_piece(self, x: int, y: int, z: int) -> None:
        self.grid[x, y, z] = EMPTY

    def get_next_empty(self, x: int, y: int) -> int:
        for z in range(BOARD_SIZE):
            if self.grid[x, y, z] == EMPTY:
                return z
        return -1

    def get_legal_moves(self, gravity: bool) -> List[Move]:
        moves: List[Move] = []
        for x, y in np.ndindex(BOARD_SIZE, BOARD_SIZE):
            if gravity:
                z = self.get_next_empty(x, y)
                if z >= 0:
                    moves.append((x, y, z))
            else:
                for z in range(BOARD_SIZE):
                    if self.grid[x, y, z] == EMPTY and (z == 0 or self.grid[x, y, z-1] != EMPTY):
                        moves.append((x, y, z))
        return moves

    def check_win(self, player: int) -> Tuple[bool, Optional[List[Move]]]:
        for line in win_lines:
            vals = self.grid[tuple(zip(*line))]
            if np.all(vals == player):
                return True, line
        return False, None

# Heuristic evaluation
weights = {1:10, 2:100, 3:1000}
def evaluate(board: Board) -> int:
    score = 0
    for line in win_lines:
        vals = board.grid[tuple(zip(*line))]
        cnt_x = np.count_nonzero(vals == X)
        cnt_o = np.count_nonzero(vals == O)
        if cnt_x and cnt_o:
            continue
        if cnt_x:
            score += weights[cnt_x]
        if cnt_o:
            score -= weights[cnt_o]
    return score

# Minimax with alpha-beta

def minimax(board: Board, depth: int, alpha: float, beta: float,
            maximizing: bool, gravity: bool) -> Tuple[float, Optional[Move]]:
    won_x, _ = board.check_win(X)
    if won_x: return math.inf, None
    won_o, _ = board.check_win(O)
    if won_o: return -math.inf, None
    if depth == 0:
        return evaluate(board), None

    moves = board.get_legal_moves(gravity)
    if not moves:
        return 0, None

    best_move: Optional[Move] = None
    if maximizing:
        value = -math.inf
        for x,y,z in moves:
            board.add_piece(X, x, y, z)
            score, _ = minimax(board, depth-1, alpha, beta, False, gravity)
            board.remove_piece(x, y, z)
            if score > value:
                value, best_move = score, (x,y,z)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value, best_move
    else:
        value = math.inf
        for x,y,z in moves:
            board.add_piece(O, x, y, z)
            score, _ = minimax(board, depth-1, alpha, beta, True, gravity)
            board.remove_piece(x, y, z)
            if score < value:
                value, best_move = score, (x,y,z)
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_move

# Immediate win/block detection
def find_forced_move(board: Board, player: int, gravity: bool) -> Optional[Move]:
    for move in board.get_legal_moves(gravity):
        board.add_piece(player, *move)
        if board.check_win(player)[0]:
            board.remove_piece(*move)
            return move
        board.remove_piece(*move)
    opp = O if player==X else X
    for move in board.get_legal_moves(gravity):
        board.add_piece(opp, *move)
        if board.check_win(opp)[0]:
            board.remove_piece(*move)
            return move
        board.remove_piece(*move)
    return None

# Display
symbols = {O:'O', X:'X', EMPTY:'.'}

def clear_screen(): os.system('cls' if os.name=='nt' else 'clear')

def display(board: Board, win_line: Optional[List[Move]] = None,
            last_move: Optional[Move] = None) -> None:
    hl_win = set(win_line) if win_line else set()
    hl_last = {last_move} if last_move else set()
    clear_screen()
    print("\n" + "="*20 + " Current Board " + "="*20)
    print("    " + "".join(f"x={x}".center(CELL_WIDTH) for x in range(BOARD_SIZE)))
    for z in reversed(range(BOARD_SIZE)):
        print(f"\nLayer {z} (z={z}):")
        for y in range(BOARD_SIZE):
            row = f"y={y:<2}"
            for x in range(BOARD_SIZE):
                sym = symbols[board.grid[x,y,z]]
                cell = format_symbol(sym)                
                if (x,y,z) in hl_last:
                    cell = color_text(cell, "93")
                elif (x,y,z) in hl_win:
                    cell = color_text(cell, "91")
                row += cell
            print(row)
    print("="*58 + "\n")

# Input parsing

def get_move(board: Board, gravity: bool) -> Optional[Move]:
    prompt = "Enter coordinates (x y) or 'u' to undo: " if gravity else "Enter coordinates (x y z) or 'u' to undo: "
    while True:
        raw = input(prompt).strip().lower()
        if raw in ('u', 'undo'):
            return None
        parts = raw.split()
        if len(parts) != (2 if gravity else 3):
            print("Wrong number of inputs.")
            continue
        try:
            coords = list(map(int, parts))
        except ValueError:
            print("Non-integer input.")
            continue
        x,y = coords[0], coords[1]
        if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
            print("x,y out of range.")
            continue
        if gravity:
            z = board.get_next_empty(x,y)
            if z < 0:
                print("Column full.")
                continue
        else:
            z = coords[2]
            if not (0 <= z < BOARD_SIZE) or (z>0 and board.grid[x,y,z-1]==EMPTY):
                print("Invalid z or floating.")
                continue
        return (x,y,z)

# Main loop

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-gravity", dest="gravity", action="store_false")
    p.add_argument("--ai", choices=["O","X"], help="AI player side")
    p.add_argument("--depth", type=int, default=2, help="Minimax depth")
    args = p.parse_args()

    board = Board()
    current = O
    last_move: Optional[Move] = None
    history: List[Tuple[int, Move]] = []
    ai_player = X if args.ai=='X' else O if args.ai=='O' else None
    game_over = False
    winning_line: Optional[List[Move]] = None  # store the winning line to highlight after game end
    p = argparse.ArgumentParser()
    p.add_argument("--no-gravity", dest="gravity", action="store_false")
    p.add_argument("--ai", choices=["O","X"], help="AI player side")
    p.add_argument("--depth", type=int, default=2, help="Minimax depth")
    args = p.parse_args()

    board = Board()
    current = O
    last_move: Optional[Move] = None
    history: List[Tuple[int, Move]] = []
    ai_player = X if args.ai=='X' else O if args.ai=='O' else None
    game_over = False

    while True:
        display(board, win_line=winning_line, last_move=last_move)

        if game_over:
            resp = input("Game over. Enter 'u' to undo last move or any key to exit: ").strip().lower()
            if resp in ('u','undo') and history:
                # Undo last move after game over
                prev_player, prev_move = history.pop()
                board.remove_piece(*prev_move)
                current = prev_player
                # If it's now AI's turn next, undo one more move
                if ai_player is not None and current == ai_player and history:
                    prev_player, prev_move = history.pop()
                    board.remove_piece(*prev_move)
                    current = prev_player
                last_move = history[-1][1] if history else None
                winning_line = None
                game_over = False
                continue
            else:
                break

        if args.ai and (ai_player==current):
            move = find_forced_move(board, current, args.gravity)
            if move is None:
                _, move = minimax(board, args.depth, -math.inf, math.inf,
                                   maximizing=(current==X), gravity=args.gravity)
            if not move:
                moves = board.get_legal_moves(args.gravity)
                move = moves[0] if moves else None
            if not move:
                print("No legal moves, draw.")
                game_over = True
                continue
        else:
            res = get_move(board, args.gravity)
            if res is None:
                if history:
                    # Undo last move
                    prev_player, prev_move = history.pop()
                    board.remove_piece(*prev_move)
                    current = prev_player
                    # If it's now AI's turn next, undo one more move
                    if ai_player is not None and current == ai_player and history:
                        prev_player, prev_move = history.pop()
                        board.remove_piece(*prev_move)
                        current = prev_player
                    last_move = history[-1][1] if history else None
                    winning_line = None
                    continue
                else:
                    print("Nothing to undo.")
                    continue
            move = res

        # Play move
        x,y,z = move
        board.add_piece(current, x,y,z)
        history.append((current, move))
        last_move = move

        # Check win/draw
        won, line = board.check_win(current)
        if won:
            winning_line = line
            display(board, win_line=winning_line, last_move=last_move)
            print(f"Player {'O' if current==O else 'X'} wins!")
            game_over = True
        else:
            # draw if no moves left
            if not board.get_legal_moves(args.gravity):
                display(board, last_move=last_move)
                print("Draw!")
                game_over = True
            else:
                current = X if current==O else O

if __name__ == '__main__':
    main()
