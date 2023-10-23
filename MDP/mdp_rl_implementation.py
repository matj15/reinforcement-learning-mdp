from copy import deepcopy
import random
import numpy as np

# return positions/states given a board (2d array)
def get_states(board):
    positions = []
    for x in range(len(board)):
        for y in range(len(board[x])):
            positions.append((x, y))
    return positions

# get reward matrix
def get_rewards(mdp, board):
    R = np.zeros(np.array(board).shape)
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            val = board[r][c]
            if board[r][c] == 'WALL':
                R[r,c] = 0.0
            else:
                R[r,c] = float(val)
    return R

# function to calculate expectation value per action
def expecation_per_action(action, s, mdp, U, R, gamma):

    # get next states for all actions
    next_states = [mdp.step(s, "UP"), mdp.step(s, "DOWN"), mdp.step(s, "RIGHT"), mdp.step(s, "LEFT")]

    probabilities = mdp.transition_function[action]
    # get utilities of the next states
    utility_next_states = [U[next_states[i][0],next_states[i][1]] for i in range(len(next_states))]
    # get rewards of the next states

    # get value of each next state multiplied with its probability
    values_w_prob_next_states = [(gamma * probabilities[i] * utility_next_states[i]) for i in range(len(next_states))]

    # sum up to get expectation value:
    expectation_action_value = sum(values_w_prob_next_states)

    return expectation_action_value


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # getting the variables we need
    U_prime = U_init[:]

    gamma = mdp.gamma
    states = get_states(mdp.board)
    #states = [(0, 3), (0,2), (0,1), (0,0), (1,3), (1,2), (1,1), (1,0), (2,3), (2,2), (2,1), (2,0)]

    U_prime = np.array(U_prime, dtype=float)

    # rewards
    R = get_rewards(mdp, mdp.board)

    max_iterations = 300
    iterations = 0
    while True:

        U = U_prime.copy()
        delta = 0.0

        for s in states:
            # actions = ["UP", "DOWN", "RIGHT", "LEFT"]
            actions = list(mdp.actions.keys())

            expectations_per_action = [expecation_per_action(actions[i], s, mdp, U, R, gamma) for i in range(len(actions))]


            max_val = max(expectations_per_action)
            action_best = expectations_per_action.index(max_val)

            U_prime[s[0],s[1]] = max_val + R[s[0],s[1]]

            if np.abs(U_prime[s[0],s[1]] - U[s[0],s[1]]) > delta:
                delta = np.abs(U_prime[s[0],s[1]] - U[s[0],s[1]])

                if delta < (epsilon*(1-gamma)/gamma):
                    return U


            # special case when we have terminal states or walls:
            if s in mdp.terminal_states:
                U_prime[s[0], s[1]] = R[s[0], s[1]]
            if mdp.board[s[0]][s[1]] == "WALL":
                U_prime[s[0], s[1]] = R[s[0], s[1]]

        # testing
        iterations +=1
        if iterations == max_iterations:
            break

    return U.tolist()
    # ========================

# function to return P(s_next | s, a)
def porbability(s_next, s, a, mdp):
    transition = mdp.transition_function #{up: probs, down: probs, right: probs, left:probs}
    # actions = ["UP", "DOWN", "RIGHT", "LEFT"]
    actions = list(mdp.actions.keys())
    # next possible states given noise
    next_states_possible = [mdp.step(s, a_i) for a_i in actions]
    # probs given selected action
    probabilities = list([transition[a]][0])

    # check if the next state is a possible state:
    if s_next in next_states_possible:

        # find the index:
        s_i = next_states_possible.index(s_next)
        # find probability
        p = probabilities[s_i]

    else:
        # state not reachable
        p = 0.0
    return float(p)

def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    states = get_states(mdp.board)
    gamma = mdp.gamma

    U = np.array(U, dtype=float)

    # policy:
    P = np.empty(U.shape).tolist()

    # rewards
    R = get_rewards(mdp, mdp.board)

    for s in states:

        # actions = ["UP", "DOWN", "RIGHT", "LEFT"]
        actions = list(mdp.actions.keys())

        expectations = []
        for a in actions:
            # iterate per action case
            next_states = [mdp.step(s,a)]

            values = []
            for s_next in next_states:
                probailitiy_val = porbability(s_next, s, a, mdp)
                inside_sum_value = probailitiy_val * (float(R[s[0], s[1]]) + gamma * float(U[s_next[0], s_next[1]]))
                values.append(inside_sum_value)
            expectation_sum = sum(values)
            expectations.append(expectation_sum)

        # maximum expectation value:
        max_val_exp = max(expectations)
        # index
        idx_max_val_exp = expectations.index(max_val_exp)
        best_action = actions[idx_max_val_exp]

        # add to policy
        P[s[0]][s[1]] = best_action

        # special cases:
        if s in mdp.terminal_states:
            P[s[0]][s[1]] = 0
        if mdp.board[s[0]][s[1]] == "WALL":
            P[s[0]][s[1]] = 'WALL'

        # P = [['UP', 'UP', 'UP', 0],
        #           ['UP', 'WALL', 'UP', 0],
        #           ['UP', 'UP', 'UP', 'UP']]

    return P

# epsilon greedy action selection
def epsilon_greedy(Q, s, epsilon, actions, state_index):
    n = random.uniform(0, 1)
    if n< epsilon:
        a = random.choice(actions)
    else:
        a_val = max(Q[state_index[s],:])
        a_index = np.argmax(Q[state_index[s],:])
        # more indices with max q val

        a = actions[a_index]

    return a

def max_a(Q, s, state_index, action_index):
    row_actions = Q[state_index[s],:].tolist()
    max_a_val = max(row_actions)
    return max_a_val


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    gamma = mdp.gamma

    actions = list(mdp.actions.keys()) # ['UP', 'DOWN', 'RIGHT', 'LEFT']

    states = get_states(mdp.board) # [(0,1), (0,2) ...]

    Q = np.zeros((len(states), len(actions)))
    alpha = learning_rate

    indexes_s = range(len(states))
    indexes_a = range(len(actions))
    state_index = dict(zip(states, indexes_s)) # {(0,1):1, (0,2):2 ...}
    action_index = dict(zip(actions,indexes_a)) # {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}


    # test with 0 rewards:
    # R = np.zeros((3,4))
    # R[0,3] = 1
    # R[1,3] = -1

    # rewards
    R = get_rewards(mdp, mdp.board)

    for episode in range(0,total_episodes):
        s = init_state
        for step in range(0, max_steps):
            while s not in mdp.terminal_states:
                a_return = epsilon_greedy(Q, s, epsilon, actions, state_index)
                a = str(a_return)
                next_state = mdp.step(s, a)

                # observe reward of the next state
                reward = R[next_state[0], next_state[1]]

                Q_S_A = Q[state_index[s],action_index[a]]
                max_a_Q_s_next_a = max_a(Q, next_state, state_index, action_index)
                Q[state_index[s],action_index[a]] = Q_S_A + alpha*(reward + gamma * max_a_Q_s_next_a - Q_S_A)

                # special cases:
                if s in mdp.terminal_states:
                    Q[state_index[s],action_index[a]] = 0.0
                if mdp.board[s[0]][s[1]] == "WALL":
                    Q[state_index[s],action_index[a]] = 0.0

                s = next_state

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

    return Q.tolist()
    # ========================

def q_table_policy_extraction(mdp, qtable):

    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #
    Q = np.array(qtable)
    P = np.empty((3,4)).tolist()

    actions = list(mdp.actions.keys()) # ['UP', 'DOWN', 'RIGHT', 'LEFT']
    states = get_states(mdp.board) # [(0,1), (0,2) ...]

    indexes_s = range(len(states))
    indexes_a = range(len(actions))
    state_index = dict(zip(states, indexes_s)) # {(0,1):1, (0,2):2 ...}
    action_index = dict(zip(actions,indexes_a)) # {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}

    for s in states:

        a_index = np.argmax(Q[state_index[s],:])

        # add to policy
        P[s[0]][s[1]] = actions[a_index]

        # special cases:
        if s in mdp.terminal_states:
            P[s[0]][s[1]] = 0
        if mdp.board[s[0]][s[1]] == "WALL":
            P[s[0]][s[1]] = 'WALL'

    return P
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
