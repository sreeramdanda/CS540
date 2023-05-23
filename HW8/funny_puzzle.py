import heapq
import math


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    for i in range(9):
        if from_state[i] != 0:
            distance += abs(i // 3 - to_state.index(from_state[i]) // 3) + abs(i % 3 - to_state.index(from_state[i]) % 3)
    return distance


def print_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def swap(state, index_1, index_2):
    """
    INPUT:
        A state (list of length 9)
        Two indices (int)
        
    RETURNS:
        A new state (list of length 9) where the two indices are swapped.
    """
    # Don't swap the zeros
    if state[index_1] == state[index_2] == 0:
        return None;

    state[index_1], state[index_2] = state[index_2], state[index_1]
    return state


def get_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []

    # Find the index of the two empty tile
    empties = []
    empties.append(state.index(0))
    empties.append(state.index(0, empties[0] + 1))

    for empty in empties:
        # if not an edge tile then swap with 4 adjacent tiles
        if empty == 4:
            succ_states.append(swap(state[:], empty, empty - 1))
            succ_states.append(swap(state[:], empty, empty + 1))
            succ_states.append(swap(state[:], empty, empty - 3))
            succ_states.append(swap(state[:], empty, empty + 3))
        # if on a corner tile then swap with 2 adjacent tiles
        elif empty == 0:
            succ_states.append(swap(state[:], empty, empty + 1))
            succ_states.append(swap(state[:], empty, empty + 3))
        elif empty == 2:
            succ_states.append(swap(state[:], empty, empty - 1))
            succ_states.append(swap(state[:], empty, empty + 3))
        elif empty == 6:
            succ_states.append(swap(state[:], empty, empty + 1))
            succ_states.append(swap(state[:], empty, empty - 3))
        elif empty == 8:
            succ_states.append(swap(state[:], empty, empty - 1))
            succ_states.append(swap(state[:], empty, empty - 3))
        # if on the middle tile on the edge then swap with 3 adjacent tiles
        elif empty == 1:
            succ_states.append(swap(state[:], empty, empty - 1))
            succ_states.append(swap(state[:], empty, empty + 1))
            succ_states.append(swap(state[:], empty, empty + 3))
        elif empty == 3:
            succ_states.append(swap(state[:], empty, empty - 3))
            succ_states.append(swap(state[:], empty, empty + 1))
            succ_states.append(swap(state[:], empty, empty + 3))
        elif empty == 5:
            succ_states.append(swap(state[:], empty, empty - 3))
            succ_states.append(swap(state[:], empty, empty - 1))
            succ_states.append(swap(state[:], empty, empty + 3))
        elif empty == 7:
            succ_states.append(swap(state[:], empty, empty - 3))
            succ_states.append(swap(state[:], empty, empty + 1))
            succ_states.append(swap(state[:], empty, empty - 1))

    while None in succ_states:
        succ_states.remove(None)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    open = []
    closed = []
    trace = []
    max_queue_len = 1

    # Add the initial state to the open list
    heapq.heappush(open, (get_manhattan_distance(state), state, (0, get_manhattan_distance(state), -1)))

    while open:
        if len(open)>max_queue_len:
            max_queue_len = len(open)

        # Get the state with the lowest h value
        cost, state, (g, h, parent) = heapq.heappop(open)

        # Add the state to the closed list
        heapq.heappush(closed, state)

        # If the state is the goal state then print the path
        if state == goal_state:
            print_statement = (state, "h={}".format(h), "moves: {}".format(cost))
            while parent != -1:
                for state_entry in trace:
                    if state_entry[1] == parent:
                        print_statement = (state_entry[1], "h={}".format(state_entry[2][1]), "moves: {}".format(state_entry[2][0])) + print_statement
                        parent = state_entry[2][2]
                        break
            
            for i in range(0, len(print_statement), 3):
                if i % 3 == 0 and i != 0:
                    print()
                print(print_statement[i], end = "")
                print(" ", end = "")
                print(print_statement[i + 1], end = "")
                print(" ", end = "")
                print(print_statement[i + 2], end = "")

            print("\nMax queue length: %d" % max_queue_len, end="")
            return
        else: 
            trace.append((cost, state, (g, h, parent)))

        # Get the successors of the state
        succ_states = get_succ(state)

        for succ_state in succ_states:
            # If the successor is not in the closed list then add it to the open list
            if succ_state not in closed:
                heapq.heappush(open, (g + 1 + get_manhattan_distance(succ_state), succ_state, (g + 1, get_manhattan_distance(succ_state), state)))


if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
