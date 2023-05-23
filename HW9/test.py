import random
import time
from game import TeekoPlayer

LOOPS = 100

if __name__ == '__main__':
    board = [
        [' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ']
    ]
    
    player = TeekoPlayer()
    total_time = 0

    for i in range(LOOPS):
        board = [
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ']
            ]
        while (player.detect_drop_phase(board)):
            board = [
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' '],
                [' ', ' ', ' ', ' ', ' ']
            ]
            piece = 'b'
            for j in range(2):
                for i in range(3):
                    # pick two random numbers between 0 and 4
                    x = random.randint(0, 4)
                    y = random.randint(0, 4)
                    # if the space is empty, place a piece there
                    if board[x][y] == ' ': board[x][y] = piece
                    # otherwise, try again
                    else: i -= 1
                piece = 'r'

        # for row in board:
        #     print(row)

        start = time.time()
        player.make_move(board)
        end = time.time()

        time_taken = end - start
        total_time += time_taken

        print("{:.2f} seconds".format(time_taken))

    print("Average time: {:.2f} seconds".format(total_time / LOOPS))