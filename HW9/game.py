import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']
    depth_limit = 2

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def succ(self, state, drop_phase):
        """ Generates all successors of the given state. During drop phase add a new piece
        of the current player's type to the board, otherwise move any of the current player's
        pieces to an unoccupied adjacent space.
        
        Args: 
            state (list of lists): current state of the game
        
        Return:
            successors (list): list of successor states
        """
        successors = []

        # During drop phase, we can add a new piece to any empty space on the board
        if drop_phase:
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        # Make a deep copy of the state
                        # Reference: https://stackoverflow.com/questions/6532881/how-to-make-a-copy-of-a-2d-array-in-python
                        successor_state = [row[:] for row in state]

                        # Update for the new state
                        successor_state[row][col] = self.my_piece
                        
                        # Add the new state to the list of successors
                        successors.append(successor_state)

        # During the move phase, we can move any of our pieces to an adjacent empty space
        else:
            for row in range(5):
                for col in range(5):

                    # If the current cell is occupied by our piece determine new states
                    if state[row][col] == self.my_piece:

                        # Check all adjacent cells for empty cells
                        # max and min are used to ensure we don't go out of bounds on edge nodes
                        for r in range(max(0, row - 1), min(5, row + 2)):
                            for c in range(max(0, col - 1), min(5, col + 2)):
                                if state[r][c] == ' ':
                                    # Make a deep copy of the state
                                    # Reference: https://stackoverflow.com/questions/6532881/how-to-make-a-copy-of-a-2d-array-in-python
                                    new_state = [row[:] for row in state]
                                    
                                    # Need to remove old piece as we are no longer in drop phase
                                    new_state[row][col] = ' '

                                    # Add old piece to new empty adjacent location
                                    new_state[r][c] = self.my_piece

                                    # Add the new state to the list of successors
                                    successors.append(new_state)

        return successors

    def heuristic_game_value(self, state):
        """ Evaluate non-terminal states using a heuristic function that labels a state between 1 and -1.

        Args:
            state (list of lists): the current state of the game

        Return:
            value (float): point between 1 and -1 representing the value of the state
                from the perspective of the player whose turn it is to move.
        """
        if self.game_value(state) != 0:
            return self.game_value(state)

        else:
            # Count the maximum number of pieces in a row for each player
            b_max = 0
            r_max = 0

            # Check rows
            for row in range(5):
                b_counter = 0
                r_counter = 0
                for col in range(5):
                    if state[row][col] == 'b':
                        b_counter += 1
                    elif state[row][col] == 'r':
                        r_counter += 1
                b_max = max(b_max, b_counter)
                r_max = max(r_max, r_counter)
            
            # Check columns
            for col in range(5):
                b_counter = 0
                r_counter = 0
                for row in range(5):
                    if state[row][col] == 'b':
                        b_counter += 1
                    elif state[row][col] == 'r':
                        r_counter += 1
                b_max = max(b_max, b_counter)
                r_max = max(r_max, r_counter)

            # Check \ diagonal
            b_counter = 0
            r_counter = 0
            for i in range(5):
                if state[i][i] == 'b':
                    b_counter += 1
                elif state[i][i] == 'r':
                    r_counter += 1
            b_max = max(b_max, b_counter)
            r_max = max(r_max, r_counter)

            # Check / diagonal
            b_counter = 0
            r_counter = 0
            for i in range(5):
                if state[i][4 - i] == 'b':
                    b_counter += 1
                elif state[i][4 - i] == 'r':
                    r_counter += 1
            b_max = max(b_max, b_counter)
            r_max = max(r_max, r_counter)

            # Check 2x2 squares
            for row in range(4):
                for col in range(4):
                    b_counter = 0
                    r_counter = 0
                    for r in range(row, row + 2):
                        for c in range(col, col + 2):
                            if state[r][c] == 'b':
                                b_counter += 1
                            elif state[r][c] == 'r':
                                r_counter += 1
                    b_max = max(b_max, b_counter)
                    r_max = max(r_max, r_counter)
            
            # Return the heuristic value
            if self.my_piece == 'b':
                return (b_max - r_max) / 8
            return (r_max - b_max) / 8

    def max_value(self, state, depth):
        """ The max-value function for the minimax algorithm

        Args:
            state (list of lists): the current state of the game
            depth (int): the current depth of the search tree

        Return:
            value (float): the value of the state
        """
        if self.game_value(state) != 0:
            return self.game_value(state)
        
        elif depth == self.depth_limit:
            return self.heuristic_game_value(state)
        
        else:
            value = float("-inf")
            for succ_state in self.succ(state, self.detect_drop_phase(state)):
                value = max(value, self.min_value(succ_state, depth + 1))
            return value

    def min_value(self, state, depth):
        """ The min-value function for the minimax algorithm

        Args:
            state (list of lists): the current state of the game
            depth (int): the current depth of the se arch tree

        Return:
            value (float): the value of the state
        """
        if self.game_value(state) != 0:
            return self.game_value(state)
        
        elif depth == self.depth_limit:
            return self.heuristic_game_value(state)
        
        else:
            value = float("inf")
            for succ_state in self.succ(state, self.detect_drop_phase(state)):
                value = min(value, self.max_value(succ_state, depth + 1))
            return value

    def detect_drop_phase(self, state):
        """ Detects whether the game is in the drop phase or not

        Args:
            state (list of lists): the current state of the game

        Return:
            drop_phase (bool): True if the game is in the drop phase, False otherwise
        """
        b_counter = 0
        r_counter = 0

        for row in state:
            for cell in row:
                if cell == 'b':
                    b_counter += 1
                elif cell == 'r':
                    r_counter += 1
        
        return b_counter + r_counter < 8

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        move = []

        drop_phase = self.detect_drop_phase(state)

        # use algorithm to find the best move
        # if not drop_phase:
        succ_states = self.succ(state, drop_phase)

        # use first succ state as defualt best move
        best_state = succ_states[0]
        best_value = self.max_value(best_state, 0)

        # compare to other succ states and pick best move based on heuristic value
        for succ_state in succ_states:
            value = self.max_value(succ_state, 0)
            if value > best_value:
                best_value = value
                best_state = succ_state
        
        # find the piece that was moved and return the new location and the old location
        new_loc = (0, 0)
        old_loc = (0, 0)
        for row in range(5):
            for col in range(5):
                if state[row][col] == ' ' and best_state[row][col] == self.my_piece:
                    new_loc = (row, col)
                elif state[row][col] == self.my_piece and best_state[row][col] == ' ':
                    old_loc = (row, col)
            
        # add the move to the beginning of the move list
        if drop_phase:
            move.insert(0, new_loc)
        else:
            move.insert(0, new_loc)
            move.insert(1, old_loc)
        
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        if state[0][0] != ' ' and state[0][0] == state[1][1] == state[2][2] == state[3][3]:
            return 1 if state[0][0] == self.my_piece else -1
        elif state[1][1] != ' ' and state[1][1] == state[2][2] == state[3][3] == state[4][4]:
            return 1 if state[1][1] == self.my_piece else -1

        # check / diagonal wins
        if state[4][0] != ' ' and state[4][0] == state[3][1] == state[2][2] == state[1][3]:
            return 1 if state[4][0] == self.my_piece else -1
        elif state[3][1] != ' ' and state[3][1] == state[2][2] == state[1][3] == state[0][4]:
            return 1 if state[3][1] == self.my_piece else -1

        # check box wins
        for row in range(3):
            for col in range(3):
                if state[row][col] != ' ' and state[row][col] == state[row][col + 1] == state[row + 1][col] == state[row + 1][col + 1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # no winner yet


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
