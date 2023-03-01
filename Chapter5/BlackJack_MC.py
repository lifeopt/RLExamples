import numpy as np
import gym
import Plot
import random

# start a new blackjack env
env = gym.make('Blackjack-v1')

# number of games to play
episodes = 500000

# sometimes you may want to discount rewards, I'm not going to cover this here
gamma = 1.

def get_epsilon(N_state_count, N_zero=100):
    """
    This is our function to calculate epsilon and is core to how we are going to pick our next action.
    
    When we first start exploring our state-action space, we have little or no knowledge of the environment, meaning we
    have little or no knowledge about what a good action might be. In this case we want to pick a random action (ie we
    want a large epsilon). As our knowledge gets better, we can have more confidence in what we're doing and so we'd like
    to pick what we know is a good action more often.
    
    We're initialising N_zero to 100, but this is a hyperparameter we can tune
    """
    return N_zero / (N_zero + N_state_count)

def get_action(Q, state, state_count, action_size):
    """
    Given our value function (Q) and state, what action should we take?
    
    If we haven't seen this state before we should pick an action at random, after all we have no information about
    what action might be best.
    
    If we have infinite experience, what should we do? In this case we would like to pick the action with the
    highest expected value all the time.
    
    To fulfil this we're going to use what is known as GLIE (Greedy in the Limit of Infinite Exploration).
    
    The idea is we pick the action with the highest expected value with a probability of `1 - epsilon` and a random
    action with probability epsilon. At first epsilon is large but it eventually decays to zero as we play an infinite
    number of games.
    
    Doing this guarentees we visit all possible states and actions.
    """
    random_action = random.randint(0, action_size - 1)
    
    best_action = np.argmax(Q[state])
        
    epsilon = get_epsilon(state_count)
    
    return np.random.choice([best_action, random_action], p=[1. - epsilon, epsilon])

def evaluate_policy(Q, episodes=10000):
    """
    Helper function which helps us evaluate how good our policy is.
    
    We do this by playing 10000 games of blackjack and returning the win ratio.
    """
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        
        done = False
        while not done:
            action = np.argmax(Q[state])
            
            state, reward, done, _ = env.step(action=action)
            
        if reward > 0:
            wins += 1
        
    return wins / episodes

def monte_carlo(gamma=1., episodes=5000, evaluate=False):

    # this is our value function, we will use it to keep track of the "value" of being in a given state
    Q = Plot.defaultdict(lambda: np.zeros(env.action_space.n))

    # to decide what action to take and calculate epsilon we need to keep track of how many times we've
    # been in a given state and how often we've taken a given action when in that state
    state_count = Plot.defaultdict(float)
    state_action_count = Plot.defaultdict(float)

    # for keeping track of our policy evaluations (we'll plot this later)
    evaluations = []

    for i in range(episodes):
        # evaluating a policy is slow going, so let's only do this every 1000 games
        if evaluate and i % 1000 == 0:
            evaluations.append(evaluate_policy(Q))
    
        # to update our value function we need to keep track of what states we were in and what actions
        # we took throughout the game
        episode = []
    
        # lets start a game!
        state = env.reset()
        done = False
        
        print("first state = ", state)
    
        # and keep playing until it's done (recall this is something Gym will tell us)
        while not done:
            # so we're in some state, let's remember we've been here and pick an action using our
            # function defined above
            state_count[state] += 1
            action = get_action(Q, state, state_count[state], env.action_space.n)

            # when we take that action, recall Gym will give us a new state, some reward and if we are done
            new_state, reward, done, _ = env.step(action=action)
        
            # save what happened, we're just going to keep the state, action and reward
            episode.append((state, action, reward))
        
            state = new_state

        # at this point the game is finished, we either won or lost
        # so we need to take what happened and update our value function
        G = 0
    
        # because you can only win or lose a game of blackjack we only get a reward at the end of the game
        # (+1 for a win, 0 for a draw, -1 for a loss). So let's start at the end of the game and work
        # backwards through our states to decide how good it was to be in a state
        for s, a, r in reversed(episode):
            new_s_a_count = state_action_count[(s, a)] + 1
            
            # we need some way of deciding how the game we just played impacted our value function. The
            # standard approach here is to take the reward(s) we got playing over multiple games and
            # taking the mean. We can update the mean as we go using what is known as incremental averaging
            # https://math.stackexchange.com/questions/106700/incremental-averageing
            G = r + gamma * G
            state_action_count[(s, a)] = new_s_a_count
            Q[s][a] = Q[s][a] + (G - Q[s][a]) / new_s_a_count
            
    return Q, evaluations

Q_mc, evaluations = monte_carlo(episodes=500000, evaluate=False)
Plot.plot_value_function(Q_mc)
