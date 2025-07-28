import ast
import numpy as np

O = 0
X = 1
empty = -1


file_path = "winning_lines_4x4x4.txt"
with open(file_path, "r") as f:
    win_lines = [ast.literal_eval(line.strip()) for line in f]

space_to_win_lines = {pos: [] for pos in np.ndindex(4, 4, 4)}
for line in win_lines:
    for pos in line:
        space_to_win_lines[pos].append(line)

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def format_symbol(symbol, highlight=False, width=5):
    raw = symbol  # The symbol without ANSI codes
    padded = f"{raw:^{width}}"  # Center raw symbol
    if highlight:
        # Apply ANSI around already-padded symbol
        return f"\033[91m{padded}\033[0m"
    return padded


class Board:
    def __init__(self):
        self.board = np.full((4, 4, 4), fill_value=-1, dtype=np.int8)

    def add_piece(self, player, x, y, z):
        self.board[x][y][z] = player

    def remove_piece(self, x, y, z):
        self.board[x][y][z] = -1

    # def display(self):
    #     for z in range(4):
    #         print(f"Layer {z}:")
    #         for y in range(4):
    #             row = ' '.join('O' if self.board[x, y, z] == O else 'X' if self.board[x, y, z] == X else '.' for x in range(4))
    #             print(row)
    #         print()

    def display(self, winning_line=None):
        symbols = {O: 'O', X: 'X', empty: '.'}
        highlight_set = set(winning_line) if winning_line else set()

        print("\n" + "="*20 + " Current Board " + "="*20)
        for z in reversed(range(4)):
            print(f"\nLayer {z} (z={z}):")
            print("     " + "  ".join(f"x={x}" for x in range(4)))
            for y in range(4):
                row = f"y={y}  "
                for x in range(4):
                    val = self.board[x, y, z]
                    symbol = symbols[val]
                    highlight = (x, y, z) in highlight_set
                    row += format_symbol(symbol, highlight)
                print(row)
        print("="*58 + "\n")


    def check_win(self, player):
        for line in win_lines:
            line_vals = np.array([self.board[x, y, z] for x, y, z in line])
            if np.all(line_vals == player):
                return True, line
        return False, None
    
    def check_win_space(self, x, y, z, player):
        for line in space_to_win_lines[(x, y, z)]:
            line_vals = np.array([self.board[x, y, z] for x, y, z in line])
            if np.all(line_vals == player):
                return True, line
        return False, None
    
    def get_next_empty(self, x, y):
        for z in range(4):
            if self.board[x, y, z] == empty:
                return z
        return -1


def main():
    board = Board()
    current_player = O
    gravity = True
    integer_input = 2 if gravity else 3  # Number of integers to input for coordinates

    while True:
        board.display()
        print(f"Player {'O' if current_player == O else 'X'}'s turn.")
        
        try:
            if gravity:
                x, y = map(int, input("Enter coordinates (x y): ").split())
                if not (0 <= x < 4 and 0 <= y < 4):
                    print("Coordinates must be between 0 and 3.")
                    continue
                z = board.get_next_empty(x, y)
                if z == -1:
                    print("Column is full. Try again.")
                    continue
            else:
                x, y, z = map(int, input("Enter coordinates (x y z): ").split())
            if not (0 <= x < 4 and 0 <= y < 4 and 0 <= z < 4):
                print("Coordinates must be between 0 and 3.")
                continue
            if board.board[x,y,z] != -1:
                print("Cell already occupied. Try again.")
                continue
            if gravity and z > 0 and board.board[x,y,z-1] == -1:
                print("You can only place a piece on top of another piece.")
                continue
        except ValueError:
            print(f"Invalid input. Please enter {integer_input} integers separated by spaces.")
            continue

        board.add_piece(current_player, x, y, z)

        has_won, win_line = board.check_win(current_player)
        if has_won:
            board.display(winning_line=win_line)
            print(f"Player {'O' if current_player == O else 'X'} wins!")
            break

        current_player = X if current_player == O else O
    


if __name__ == "__main__":
    main()
