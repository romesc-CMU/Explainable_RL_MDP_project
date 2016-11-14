from __future__ import division 
import operator, math, random, copy, sys, os.path, bisect
from IPython import embed

def if_(test, result, alternative):
    """Like C++ and Java's (test ? result : alternative), except
    both result and alternative are always evaluated. However, if
    either evaluates to a function, it is applied to the empty arglist,
    so you can delay execution by putting it in a lambda.
    >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
    'ok'
    """
    if test:
        if callable(result): return result()
        return result
    else:
        if callable(alternative): return alternative()
        return alternative

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        # if x_score < best_score:
            # best, best_score = [x], x_score
        # elif x_score == best_score:
            # best.append(x)
        if best_score-x_score>10**(-5):
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x))
def vector_add(a, b):
    """Component-wise addition of two vectors.
    >>> vector_add((0, 1), (8, 9))
    (8, 10)
    """
    return tuple(map(operator.add, a, b))

# (1,0) = south
orientations = [(-1,0), (1,0), (0,-1), (0,1)]
def turn_right(orientation):
    if orientation == (0,1):
        return (1,0)
    elif orientation == (0,-1):
        return (-1,0)
    elif orientation == (1,0):
        return (0,-1)
    elif orientation == (-1,0):
        return (0,1)
    # return orientations[orientations.index(orientation)-1]
def turn_left(orientation):
    if orientation == (0,1):
        return (-1,0)
    elif orientation == (0,-1):
        return (1,0)
    elif orientation == (1,0):
        return (0,1)
    elif orientation == (-1,0):
        return (0,-1)
    # return orientations[(orientations.index(orientation)+1) % len(orientations)]
def update(x, **entries):
    """
    Update a dict; or an object with slots; according to entries.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)   
    else:
        x.__dict__.update(entries) 
    return x 

"""
the grid is in [row, col]
In the future, everything, including action and state, is all in [row,col]
and the row 0 is on the top, like an 2D array [[1,2,3],[4,5,6]]
"""

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    # XXX: gamma = discount
    # XXX: terminal = the end state
    #      (when we get to the end state, whatever reward we have, we will stop moving)
    # TODO: now R is a function of s, but it could be a function of (s,a)
    def __init__(self, actlist, terminals, gamma=.9):
        update(self, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminals, gamma=0.9):
        MDP.__init__(self, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        # store rewards in a map
        for i in range(self.rows):
            for j in range(self.cols):
                # reward = {(0, 0): -0.04,}   
                self.reward[i, j] = grid[i][j]
                if grid[i][j] is not None:
                    # if the state is reachable, add it
                    self.states.add((i, j))

    def T(self, state, action):
        # [(probability, end_state),]
        if action == None:
            # XXX: should be 1.0
            return [(0.0, state)]
        else:
            # return [(0.8, self.go(state, action)),
                    # (0.1, self.go(state, turn_right(action))),
                    # (0.1, self.go(state, turn_left(action)))]
            # XXX: no uncertainty
            return [(1., self.go(state, action))]


    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        # if the next state is not valid, then don't go there 
        return if_(state1 in self.states, state1, state)

    # transform a grid_map into a arrow_grid_map based on the policy in the mapping
    def to_grid(self, mapping):
        """Convert a mapping from (i, j) to v into a [[..., v, ...]] grid."""
        return [[mapping.get((i,j), None) for j in range(self.cols)] 
                for i in range(self.rows)]

    # transform from a policy {[state_i,state_j]:[action_i,action_j]}
    # to a path in the grid {[i,j]:'>'}
    def to_arrows(self, policy):
        chars = {(0,1):'>', (-1,0):'^', (0,-1):'<', (1,0):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))

# use value_iteration to learn the utility function for each position in grid
def value_iteration(mdp, epsilon=0.001):
    utilities = []
    "Solving an MDP by value iteration. [Fig. 17.4]"
    # value of each state
    U1 = []
    for s in mdp.reward:
        U1.append((s,None))
    U1 = dict(U1)
    for s in mdp.states:
        U1[s] = 0.
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    counter = 0
    goal_cost = 0.
    prev_goal_cost = 0.
    while True:
        U = U1.copy()
        delta = 0.
        max_s = None
        for s in mdp.states:
            # XXX: if s is obstacle
            if mdp.reward[s] != None:
                # XXX: if s is terminal state
                if mdp.actions(s) != None:
                    # XXX: R(s) is the current position
                    U1[s] = R(s) + gamma*max([sum([p*U[s1] for (p, s1) in T(s, a)])
                                                for a in mdp.actions(s)])
                    # not working
                    # U1[s] = gamma*max([sum([p*(R(s1)+U[s1]) for (p, s1) in T(s, a)])
                                                # for a in mdp.actions(s)])

                    # print 'for state ',s,' ',U[s],' => ',U1[s], ', R=',R(s) 
                    if delta<abs(U1[s] - U[s]):
                        max_s = s
                    delta = max(delta, abs(U1[s] - U[s]))
                else:
                    # XXX: For terminal state, utility = reward at that position
                    U1[s] = R(s)
            else:
                U1[s] = None
        utilities.append(U1.copy())
        # when the max change is very small (convergence), then we should stop
        # print epsilon * (1 - gamma) / gamma = 1*10^-5

        #---------------------------------------------------------
        """

        # print delta - epsilon * (1. - 0.01) * 0.01
        # print counter
        counter += 1
        if counter > 100:
            exit(0)
        if counter >= 0:
        # if delta - epsilon * (1. - 0.01) * 0.01 < 20:
            # cycle detecting
            utility = utilities[-1]
            act_map = []
            for y in range(10):
                row = []
                for x in range(20):
                    row.append('.')
                act_map.append(row)

            for y in range(10):
                for x in range(20):
                    max_act = None
                    max_uti = -float('Inf')
                    if (y-1,x) in utility:
                        if utility[(y-1,x)]>max_uti:
                            max_uti = utility[(y-1,x)]
                            max_act = '^'
                    if (y+1,x) in utility:
                        if utility[(y+1,x)]>max_uti:
                            max_uti = utility[(y+1,x)]
                            max_act = 'v'
                    if (y,x-1) in utility:
                        if utility[(y,x-1)]>max_uti:
                            max_uti = utility[(y,x-1)]
                            max_act = '<'
                    if (y,x+1) in utility:
                        if utility[(y,x+1)]>max_uti:
                            max_uti = utility[(y,x+1)]
                            max_act = '>'
                    act_map[y][x] = max_act
            for i in act_map:
                print i
            rew_map = []
            for y in range(5,10):
                row = []
                for x in range(1,6):
                    row.append('%2s' % mdp.reward[(y,x)])
                rew_map.append(row)
            for i in rew_map:
                print i
            print 
            utility_map = []
            for y in range(5,10):
                row = []
                for x in range(1,6):
                    row.append('%2s' % utilities[-1][(y,x)])
                utility_map.append(row)
            for i in utility_map:
                print i
            print 
#             utility_map = []
            # for y in range(10):
                # row = []
                # for x in range(20):
                    # row.append('%2s' % utilities[-2][(y,x)])
                # utility_map.append(row)
            # for i in utility_map:
                # print i
            
            print max_s 
            print '----------------------------------'
            # embed()
            # exit(0)
 
        """
        #---------------------------------------------------------

        # if delta < epsilon * (1 - gamma) / gamma:
        # if delta < epsilon * (1. - gamma) * gamma:

        # if delta < epsilon * (1. - 0.01) * 0.01:
        if delta < epsilon * (1. - 0.01) * 0.01:
            # exit(0)
            return utilities[-1]

# iterate through the grid and get the best solution based on the utility function
"""
pi = {state: action,}
pi =
{(0, 0): (0, 1),
 (0, 1): (0, 1),
 (0, 2): (1, 0),
 (1, 0): (1, 0),
 (1, 2): (1, 0),
 (2, 0): (0, 1),
 (2, 1): (0, 1),
 (2, 2): (1, 0),
 (3, 0): (-1, 0),
 (3, 1): None,
 (3, 2): None}
 """
def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax_list(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    # return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])
    # https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node20.html
    return mdp.R(s)+sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_iteration(mdp, init_pi):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    # value of each state
    U1 = []
    for s in mdp.reward:
        U1.append((s,None))
    U1 = dict(U1)
    for s in mdp.states:
        U1[s] = 0.
    pi = init_pi
    pis = [copy.deepcopy(pi)]
    while True:
        U = U1.copy()
        U1 = policy_evaluation_(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            actions = argmax_list(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
            # break ties by NSWE order
            # north
            if (-1,0) in actions:
                a = (-1,0)
            elif (1,0) in actions:
                a = (1,0)
            elif (0,-1) in actions:
                a = (0,-1)
            elif (0,1) in actions:
                a = (0,1)
            else:
                a = None
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        pis.append(copy.deepcopy(pi))
        print U1
        if unchanged:
            return U1

# update the utility for k times based on the policy
def policy_evaluation_(pi, U1, mdp, epsilon=0.001):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0.
        for s in mdp.states:
            # XXX: if s is obstacle
            if mdp.reward[s] != None:
                # XXX: if s is terminal state
                if mdp.actions(s) != None:
                    # XXX: R(s) is the current position
                    U1[s] = R(s) + gamma * sum([p * U[s1] for (p, s1) in T(s, pi[s])])
                    delta = max(delta, abs(U1[s] - U[s]))
                else:
                    # XXX: For terminal state, utility = reward at that position
                    U1[s] = R(s)
            else:
                U1[s] = None
        # when the max change is very small (convergence), then we should stop
        # print epsilon * (1 - gamma) / gamma = 1*10^-5
        # if delta < epsilon * (1 - gamma) / gamma:
        # if delta < epsilon * (1. - gamma) * gamma:
        print delta - epsilon * (1. - 0.01) * 0.01
        if delta < epsilon * (1. - 0.01) * 0.01:
             return U
    return U

################################################################################
# Parameters:
#   rewards is a 2d array that contains the rewards for each state (None for walls)
#   ware is the list of warehouse locations
#   nofly is the list of no-fly-zone locations
# 
# Return Value:
#   a tuple (value_hist, policy)
#   value_hist is a list of 2d arrays corresponding to the value of each 
#       state after each iteration (Start after the first iteration when values = rewards) 
#   policy is a 2d array corresponding to the optimal policy for each state with
#       values of None for walls, warehouses, and no-fly-zones. All others have a 
#        value from ('N','S','W','E')
################################################################################
# def value_iteration_example(rewards, ware, nofly):
    # ware.extend(nofly)
    # m = GridMDP(rewards, terminals=ware)
    # utilities = value_iteration(m)
    # ret_utilities = []
    # ret = []
    # for y in range(m.rows):
        # row = []
        # for x in range(m.cols):
            # row.append(None)
        # ret.append(row)
    # for u in utilities:
        # tmp = copy.deepcopy(ret)
        # for k,v in u.iteritems():
            # tmp[k[0]][k[1]] = v
        # ret_utilities.append(tmp)

    # ret_actions = copy.deepcopy(ret)
    # pi = best_policy(m,utilities[-1])
    # for k,v in pi.iteritems():
        # # break ties by NSWE order
        # # north
        # if (-1,0) in v:
            # x = 'N'
        # elif (1,0) in v:
            # x = 'S'
        # elif (0,-1) in v:
            # x = 'W'
        # elif (0,1) in v:
            # x = 'E'
        # else:
            # x = None
        # ret_actions[k[0]][k[1]] = x
    # return (ret_utilities,ret_actions)
# print value_iteration_example([[0, -1], [0, 1]], [(1, 1)], [(0, 1)])
# print value_iteration_example([[-0.04, -0.04, 1, None], [-0.01, -0.01, -0.2, 5], [None, -0.1, -5, -0.04], [3, -0.1, -3, -0.2]], [(0, 2), (1, 3), (3, 0)], [(2, 2), (3, 2)])
# print value_iteration_example([[-0.1, None, -0.1, -5, 3], [-0.1, -1, -0.1, -0.1, -0.1], [2, -4, None, -0.1, 1], [-1, -2, -0.1, -0.1, -0.1], [-0.2, None, 1, -0.1, -0.1]],[(0, 4), (2, 4), (4, 2)],[(0, 3), (1, 1), (2, 1), (3, 0), (3, 1)])
# exit(0)




################################################################################
# Parameters:
#   rewards is a 2d array that contains the rewards for each state (None for walls)
#   ware is the list of warehouse locations
#   nofly is the list of no-fly-zone locations
# 
# Return Value:
#   a tuple (policy_hist, policy)
#   policy_hist is a list of 2d arrays corresponding to the policy following each iteration 
#   policy is a 2d array corresponding to the optimal policy for each state with
#       values of None for walls, warehouses, and no-fly-zones. All others have a 
#        value from ('N','S','W','E')
################################################################################
# def policy_iteration_example(rewards, ware, nofly):
    # ware.extend(nofly)
    # m = GridMDP(rewards, terminals=ware)
    # # random.seed()
    # # init_pi = dict([(s, random.choice(m.actions(s))) for s in m.states])
    # init_pi = []
    # for s in m.states:
        # if m.actions(s) != [None]:
            # init_pi.append((s,(-1.,0,)))
        # else:
            # init_pi.append((s,None))
    # init_pi = dict(init_pi)
    # pis = policy_iteration_(m, init_pi)

    # rets = []
    # ret = []
    # for y in range(m.rows):
        # row = []
        # for x in range(m.cols):
            # row.append(None)
        # ret.append(row)
    # for pi in pis:
        # tmp = copy.deepcopy(ret)
        # for k,v in pi.iteritems():
            # # north
            # if (-1,0) == v:
                # x = 'N'
            # elif (1,0) == v:
                # x = 'S'
            # elif (0,-1) == v:
                # x = 'W'
            # elif (0,1) == v:
                # x = 'E'
            # else:
                # x = None
            # tmp[k[0]][k[1]] = x
        # rets.append(tmp)
    # return (rets,rets[-1])
# print policy_iteration_example([[0, -1], [0, 1]], [(1, 1)], [(0, 1)])
# print policy_iteration_example([[-0.04, -0.04, 1, None], [-0.01, -0.01, -0.2, 5], [None, -0.1, -5, -0.04], [3, -0.1, -3, -0.2]], [(0, 2), (1, 3), (3, 0)], [(2, 2), (3, 2)])
# exit(0)


def value_iteration_wrapper(reward_map, terminals):
    # example not working when gamma == 1.
    # reward_map = [[3,-1,2],[2,-3,2],[-3,-1,10]]
    # example working when gamma == 1.
    # reward_map = [[-3,-1,-2],[-2,-3,-2],[-3,-1,10]]
    # terminals = [(2,2)]


    # policy iteration
    # m = GridMDP(reward_map, terminals,gamma=0.99)
    # init_pi = []
    # for s in m.states:
        # if m.actions(s) != [None]:
            # init_pi.append((s,(0,1,)))
        # else:
            # init_pi.append((s,None))
    # init_pi = dict(init_pi)
    # utility = policy_iteration(m, init_pi)
    # embed()

    # value iteration
    m = GridMDP(reward_map, terminals,gamma=1.0)
    utility = value_iteration(m)

    ret = []
    for y in range(m.rows):
        row = []
        for x in range(m.cols):
            row.append(None)
        ret.append(row)

    ret_utility = []
    ret_utility = copy.deepcopy(ret)
    for k,v in utility.iteritems():
        ret_utility[k[0]][k[1]] = v

    ret_actions = copy.deepcopy(ret)
    pi = best_policy(m,utility)
    for k,v in pi.iteritems():
        x = []
        # we add all the optimal actions
        # break ties by NSWE order
        # north
        if (-1,0) in v:
            x.append('^')
        if (1,0) in v:
            x.append('v')
        if (0,-1) in v:
            x.append('<')
        if (0,1) in v:
            x.append('>')
        ret_actions[k[0]][k[1]] = x
    return (ret_utility,ret_actions,m)



def mdp_path_generation(reward_map, terminals, starting_pt_row, starting_pt_col,\
        ending_pt_row, ending_pt_col):
    utility_map, opt_action_map, mm = value_iteration_wrapper(reward_map,terminals)

    # Traverse the multiple solutions using BFS (state = entire path)
    i = starting_pt_row
    j = starting_pt_col

    num_col = len(reward_map[0])
    num_row = len(reward_map)
    queue = []
    paths = ()
    queue.append(copy.deepcopy(((starting_pt_row,starting_pt_col),)))

    while len(queue) > 0:
        cur_path = copy.deepcopy(queue.pop(0))
        prev_waypt = cur_path[-1]
        # done
        if prev_waypt[0] == ending_pt_row and prev_waypt[1] == ending_pt_col:
            paths += (copy.deepcopy(cur_path), )
            continue
        for action in opt_action_map[prev_waypt[0]][prev_waypt[1]]:
            # XXX: Assuming there is no uncertainty
            if action == 'v':
                action_coord = (1,0)
            elif action == '^':
                action_coord = (-1,0)
            elif action == '<':
                action_coord = (0,-1)
            elif action == '>':
                action_coord = (0,1)
            next_waypt = mm.T(prev_waypt,action_coord)[0][1]

            # avoid cycle
            if next_waypt not in cur_path:
                tmp = copy.deepcopy(cur_path)
                tmp += (next_waypt,)
                queue.append(tmp)
    # print len(paths)
    assert(len(set(paths))==len(paths))
    for p in paths:
        assert(len(p)==len(set(p)))

    # embed()
    return utility_map, opt_action_map, paths
        

if __name__ == "__main__":
    utility_map, opt_action_map, paths = \
            mdp_path_generation([[-1,-1,-1], [-1,-1,-1], [-1,-1,10]], [(2, 2)], 0,0,2,2)
    embed()


