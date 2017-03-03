from __future__ import print_function
import time
import curses
import random

BOARD_SIZE = 10
UPDATE_COUNT = 100


class RandomField:
    def __init__(self):
        self.val = 0

    def update_val(self):
        self.val = random.randint(10,99)

    def get_val(self):
        return self.val

class RandomBoard:
    def __init__(self, board_size):
        self.board = []
        self.board_size = board_size
        for x_ind in range(0, board_size):
            self.board.append([])
            for y_ind in range(0, board_size):
                self.board[x_ind].append(RandomField())

    def update(self):
        for line in self.board:
            for field in line:
                field.update_val()

    def print_board(self):
        for x_ind in range(0, self.board_size):
            for y_ind in range(0, self.board_size):
                stdscr.addstr(x_ind, y_ind, str(self.board[x_ind][y_ind].get_val()))

if __name__ == '__main__':
    random_board = RandomBoard(BOARD_SIZE)
    stdscr = curses.initscr()
    for i in range(0, UPDATE_COUNT):
        random_board.update()
        random_board.print_board()
        time.sleep(0.1)
        stdscr.refresh()

    random_board.update()
    random_board.print_board()
    curses.endwin()
    print("\nThat's all folks")


