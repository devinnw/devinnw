import random
import math


BOT_NAME = "Deputy Hawk" #+ 19

DOPTH = 0


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


def maximizer(state):
    if state.is_full():
        return state.utility()
    v = -1000000000
    for a, s in state.successors():
        v = max(v, minimizer(s))
    return v


def minimizer(state):
    if state.is_full():
        return state.utility()
    v = 10000000000
    for a, s in state.successors():
        v = min(v, maximizer(s))
    return v


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        #
        # Fill this in!
        #

        if state.is_full():
            return state.utility()
        if (state.next_player() < 0):
            for a, s in state.successors():
                return minimizer(state)
        elif (state.next_player() > 0):
            for a, s in state.successors():
                return maximizer(state)

        return -1


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        #
        # Fill this in!
        #

        values = []

        if (self.depth_limit == 0) | (state.is_full()):
            return self.evaluation(state)
        else:
            for x, child in state.successors():
                if self.depth_limit is None:
                    values.append(self.minimax(child))
                else:
                    self.depth_limit -= 1
                    values.append(self.minimax(child))
                    self.depth_limit += 1

        if state.next_player() > 0:
            return max(values)

        return min(values)

    # a modified version of streaks that introduces new weights
    # first, it checks if a streak is separated by one 0. in this case, it will continue that previous len.
    # moreover, zeros are rated higher. if a zero comes after a streak, the length is not reset.
    # finally, it only checks in the current player's favor.
    def modstreaks(self, lst, playa):
        """Get the lengths of all the streaks of the same element in a sequence."""
        rets = []  # list of (element, length) tuples
        superprev = lst[0]
        prev = lst[0]
        gone = 0
        curr_len = 0
        prev_len = 0
        for curr in lst[1:]:
            if curr == playa & prev == 0 & superprev == playa & gone > 2:
                curr_len += prev_len + 1
            elif curr == playa & curr == prev:
                curr_len += 1
            elif curr == 0 & prev == playa:
                curr_len += 0
            else:
                rets.append((prev, curr_len))
                prev_len = curr_len
                curr_len = 1
            superprev = prev
            prev = curr
            gone += 1
        rets.append((prev, curr_len))
        return rets

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        #
        # Fill this in!
        #
        if state.next_player() < 0:
            current = 1
        else:
            current = -1

        heuristic = 0
        evil_heuristic = 0

        # modified for loops to get rows, cols then pass to streaks
        # only checks the current player
        # weighs any sort of streak by length onky, in part due to weight modifications in modstreaks.
        for run in state.get_rows() + state.get_cols() + state.get_diags():
            for elt, length in self.modstreaks(run, current):
                heuristic += length

        for run in state.get_rows() + state.get_cols() + state.get_diags():
            for elt, length in self.modstreaks(run, current * -1):
                evil_heuristic += length

        return heuristic - evil_heuristic

# state, a = our alpha and b is our beta.
def prunemaximizer(state, a, b):
    if state.is_full():
        return state.utility()
    v = -1000000000
    for x, s in state.successors():
        v = max(v, pruneminimizer(s, a, b))
        if v >= b:
            return v
        a = max(a, v)
    return v


def pruneminimizer(state, a, b):
    if state.is_full():
        return state.utility()
    v = 100000000
    for x, s in state.successors():
        v = min(v, prunemaximizer(s, a, b))
        if v <= a:
            return v
        b = min(b, v)
    return v


class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""

    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        #
        # Fill this in!
        #

        if state.is_full():
            return state.utility()
        if (state.next_player() < 0):
            for a, s in state.successors():
                return pruneminimizer(state, -10000000, 1000000)
        elif (state.next_player() > 0):
            for a, s in state.successors():
                return prunemaximizer(state, -10000000, 1000000)

        return 13  # Change this line!


# N.B.: The following class is provided for convenience only; you do not need to implement it!

class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 26  # Change this line, unless you have something better to do.

