from heapq import *
import math
import time
import argparse
import PIL
import PIL.Image
from collections import namedtuple

#
#  You will need to install the python package Pillow for this to work
#  generally, on a python install you can do that with the command
#  > pip install pillow
#  (issued from the command line)
#
#  Once installed, you should be able to import PIL
#  If you can't, make sure that your pip is from the same python install
#  as the python 3 that you're using.  You can run
#
# > pip -V
#
# to determine the version of pip, but more importantly, the python
# that this pip is attached to.
#
# you want to run the pip that is attached to the same install of python
# as the python you're using (determinable via the command `which python3`)
#  https://pypi.org/project/Pillow/
#

# A Search Tree Node (TNode) is a tuple of (f,g,h,state,parent)
# where the parent value is a reference to another Search Tree Node
# of the same form.  The root of the search tree is represents the
# initial state and has no parent.
#
# The line below creates a new *class* called "TNode" that has
# 5 slots that can be accessed with the names:
#  'f', 'g', 'h', 'state' and 'parent' respectively.
# Since TNodes are tuples (but with names), they are immutable
# and all values must be supplied at initialization.
# Take a look at the code segments below to see how these objects
# are created and how instance data is accessed.
#
# 'f', 'g', and 'h' and represent the elements of the path cost
# 'f' is the total cost of the path (refered to as f in 3rd edition,
#     and as 'path cost' in 4th edition)
# 'g' is the path cost incurred so far (for a goal state f == g,
#     but for a non-goal state, g indicates how much it costs to
#     get to the specified state)
# 'h' is the heuristic cost, or the approximate cost remaining
#     to be incurred on the path to the goal. For a goal state,
#     there is no cost remaining, so h == 0. In A* the h-value
#     helps prioritize some nodes over others.
#
# In A* recall that f = g + h
# In Uniform Cost Search (ala Dijkstra's) f = g
TNode = namedtuple('TNode', ['f', 'g', 'h', 'state', 'parent'])


class SearchProblem():
    """An absract class representing a search problem."""

    def successors(self, state):
        """The successor function: returns a generator of sucessor states and 
        their edge costs as (successor, edge cost) pairs."""
        pass

    def is_goal(self, state):
        """returns True iff the specified state satisifies the goal condition"""
        pass


class GridProblem(SearchProblem):
    """A Grid Problem allows cardinal direction movement on an 2D grid.
    The grid is represented with a list of lists and edge weights are the
    average value of the two adjacent tiles"""

    def __init__(self, listoflists=None, goaltest=None, hfn=None):
        """Setup an instance of a GridProblem.
        States in a GridProblem are represented as (x,y) tuples
        and represent a particular location in the grid.

        Arguments:
            listoflists: a 2-d grid, represented as a list-of-lists, or
                         None in which case a small grid is used.
            goaltest:  a function which takes a state and returns
                         True iff the state satisifies the goaltest

            hfn:       a heuristic function which takes a state
                         and estimates the distance remaining to the goal.
        """
        if listoflists:
            self.grid = listoflists
        else:
            self.grid = [[1, 10, 10, 10, 10],
                         [1, 1, 10, 10, 10],
                         [10, 1, 1, 1, 1],
                         [10, 10, 1, 10, 1],
                         [10, 10, 1, 10, 1]]

        if goaltest:
            self.is_goal = goaltest
        else:
            self.is_goal = lambda x: x == (4, 4)

        if hfn:
            self.h = hfn
        else:
            self.h = lambda x: abs(x[0] - 4) + abs(x[1] - 4)

    def successors(self, state):
        """In a GridProblem, successor states are vertically
        or horizontally adjacent. The edge weight is the
        average of the two tiles."""

        if state[0] > 0:
            ns = (state[0] - 1, state[1])  # left
            yield (ns, (self.grid[state[0]][state[1]] +
                        self.grid[ns[0]][ns[1]]) / 2.0)

        if state[0] < len(self.grid) - 1:
            ns = (state[0] + 1, state[1])  # right
            yield (ns, (self.grid[state[0]][state[1]] +
                        self.grid[ns[0]][ns[1]]) / 2.0)

        if state[1] > 0:
            ns = (state[0], state[1] - 1)
            yield (ns, (self.grid[state[0]][state[1]] +
                        self.grid[ns[0]][ns[1]]) / 2.0)

        if state[1] < len(self.grid[0]) - 1:
            ns = (state[0], state[1] + 1)
            yield (ns, (self.grid[state[0]][state[1]] +
                        self.grid[ns[0]][ns[1]]) / 2.0)


class ImageProblem(SearchProblem):
    """An ImageProblem allows cardinal direction movement on an x-y grid.
    Here, the grid is represented with pixels from the image. A state
    is represented by an (x,y) tuple where 0 <= x < image.width and
    0 <= y < image.height. Edges connect states in cardinal directions,
    AND diagonals."""

    def __init__(self, imagepth, goaltest, hfn=None):
        self.img = PIL.Image.open(imagepth)
        self.is_goal = goaltest
        self.h = hfn

    def successors(self, state):
        """_ Part 1:  Implement This Method _

        Hint: Use self.img.getpixel(p) to get a tuple of RGB
        values; p should be an (x,y) coordinate tuple specifying the
        pixel location you wish to obtain RBG vaues for.

        the cost between two adjacent nodes should be calculated as:
         1 + sum( ((pixel value difference in channel)/32)**2 for each channel R,G,B )

        that is,
        (1) find the difference in each R,G,B channel,
        (2) divide those differences by 32 (to compress them a bit)
        (3) square the result (to make it positive and non-linear)
        (4) sum across the channels (R,G and B).

        so for two states whose R,G,B values are (100,100,100) and (100,116,36)
        the cost should be 5.25 = (1 + (0/32)**2 + (16/32)**2, (64/32)**2).

        Yields all successors of the specified state as
        tuples of: (nextstate, edgecost)
        """

        def get_cost(state1, state2):
            tuple1 = self.img.getpixel(state1)
            tuple2 = self.img.getpixel(state2)
            r = (abs(tuple1[0] - tuple2[0]) / 32) ** 2
            g = (abs(tuple1[1] - tuple2[1]) / 32) ** 2
            b = (abs(tuple1[2] - tuple2[2]) / 32) ** 2
            return 1 + r + g + b

        def in_img_range(x, y):  # return True when given coordinate is in the range of the image
            return 0 <= x < self.img.width and 0 <= y < self.img.height

        if state[0] > 0:  # move to the left / West
            next_state = (state[0] - 1, state[1])
            yield next_state, get_cost(state, next_state)

        if state[0] < self.img.width - 1 :  # move to the right / East
            next_state = (state[0] + 1, state[1])
            yield next_state, get_cost(state, next_state)

        if state[1] > 0:  # move downward / to the South
            next_state = (state[0], state[1] - 1)
            yield next_state, get_cost(state, next_state)

        if state[1] < self.img.height - 1:  # move upward / to the North
            next_state = (state[0], state[1] + 1)
            yield next_state, get_cost(state, next_state)

        if in_img_range(state[0] - 1, state[1] + 1):  # diagonal of NorthWest (left on x direction, up on y direction)
            next_state = (state[0] - 1, state[1] + 1)
            yield next_state, get_cost(state, next_state)

        if in_img_range(state[0] - 1, state[1] - 1):  # diagonal of SouthWest (left on x direction, down on y direction)
            next_state = (state[0] - 1, state[1] - 1)
            yield next_state, get_cost(state, next_state)

        if in_img_range(state[0] + 1, state[1] + 1):  # diagonal of NorthEast (right on x direction, up on y direction)
            next_state = (state[0] + 1, state[1] + 1)
            yield next_state, get_cost(state, next_state)

        if in_img_range(state[0] + 1, state[1] - 1):  # diagonal of SouthEast(right on x direction, down on y direction)
            next_state = (state[0] + 1, state[1] - 1)
            yield next_state, get_cost(state, next_state)

    def show_path(self, listofstates, pathfname='path.png'):
        """Given a list of states (each state specified as a tuple of
        (x,y) coordinates within the image, paint all states red and
        save the image."""

        imgcopy = self.img.copy()
        for state in listofstates:
            imgcopy.putpixel(state, (255, 0, 0))
        imgcopy.save(pathfname)


class AStar():

    def __init__(self, problem):
        """Initializer for an AStar search method.

        You must maintain the following instance variables:
          (1) self.reached:  a lookup table (dict) that maps states to TNode
                 instances. (also refered to as a closed list)
          (2) self.frontier:   a priority queue containing TNode instances
                               (also refered to as an open list)

        """

        self.problem = problem
        self.reached = {}
        self.frontier = []

    def search(self, initialstate, costlimit=None, quiet=True):
        """_ Part 2: Implement This Method _

        Performs A* Search.
         (1) initializes frontier and reached  instance variables
         (2) performs A* seach using self.problem to obtain:
              a successor function, heuristic function, and goal test
         (3) returns a search tree node (i.e., path), or False

        Arguments:
          costlimit - None (no limit) or a value such that the search
                       will quit when a node is removed from the frontier
                       with g > costlimit

          quiet - prints no output if this is True

        Returns a TNode constituting a path, or False if the search failed.
        """

        def shortest_distance_tnode(frontier):  # a function to get the shortest TNode and remove from self.frontier
            heapify(frontier)
            return heappop(frontier)

        # initializes frontier and reached  instance variables
        current_state = TNode(0, 0, 0, initialstate, None)
        self.frontier.append(current_state)  # add the initial state to the frontier

        while len(self.frontier) > 0:
            current_state = shortest_distance_tnode(self.frontier)  # get the state that cost the least in the frontier

            if self.problem.is_goal(current_state[3]):  # if we found the goal/final-state
                if not quiet:  # print output
                    self.problem.show_path(listofstates=self.list_of_states(current_state))
                return current_state

            for successor in self.problem.successors(current_state[3]):  # loop through each successor of current state
                new_state = successor[0]
                path_cost = successor[1] + current_state[1]  # the sum of the successor's and its parent's path cost
                exist_tnode = self.reached.get(new_state)  # check if the successor already in the reached{}
                h_cost = self.problem.h(new_state)  # calculate the heuristic cost with the successor state

                # if the successor does not exist in the reach{} TNode
                # or, we found smaller path cost for the state of the successor
                if exist_tnode is None or path_cost < exist_tnode[1]:
                    # create new TNode with associated values of the successor
                    new_tnode = TNode(path_cost + h_cost, path_cost, h_cost, new_state, current_state)
                    self.reached[new_state] = new_tnode  # update the dict-reached{}
                    self.frontier.append(new_tnode)  # add the successor to the frontier
        return False

    def list_of_states(self, treenode):
        """Given a TNode instance, create a list of states representing
        the path *from the root* of the tree to the specified TNode instance.
        This should be obtainable by following the parent references from the
        given TNode back to the root.

        Returns a list of states, the first state should be an initial state,
        the last state should be the state represented by the specified TNode
        instance"""

        stack = []
        node = treenode
        while node:
            (f, g, h, s, parent) = node
            stack.append(TNode(f, g, h, s, None))
            node = parent
        stack.reverse()
        return [s.state for s in stack]


if __name__ == "__main__":
    # You shouldn't need to modify this...

    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs="?")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-x', type=int, default=0,
                        help="initial x coordinate (default 0)")
    parser.add_argument('-y', type=int, default=0,
                        help="initial y coordinate (default 0)")
    parser.add_argument('-n', type=int,
                        help="show only first n states on the path",
                        default=-1)

    args = parser.parse_args()

    if args.image:
        image = PIL.Image.open(args.image)
        size = (image.width, image.height)


        # This heuristic always works!
        def h_fn(x):
            return 0


        def goal_fn(x):
            return x == (size[0] - 1, size[1] - 1)


        prob = ImageProblem(args.image, goal_fn, h_fn)
    else:
        prob = GridProblem()

    a = AStar(prob)
    start = time.time()

    # WARNING: the unit tests will run your program from the command line
    # and look at the output.  Make sure your search obeys the 'quiet' flag.
    # and make sure you you don't change this when you submit.
    tnode = a.search((args.x, args.y), quiet=(not args.verbose))
    stop = time.time()
    states_on_path = a.list_of_states(tnode)
    left_to_print = args.n
    for s in states_on_path:
        if left_to_print == 0:
            break
        print(s)
        left_to_print -= 1

    print("Path found in %.2f seconds" % (stop - start))
    print("%d states in the reached dict" % len(a.reached))
    print("%d states on path to goal" % len(states_on_path))
    print("cost is: %.2f" % tnode.g)

    if args.image:
        prob.show_path(states_on_path)
