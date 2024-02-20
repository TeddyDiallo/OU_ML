# The states and actions from the provided MDP
states = ["Mount Olympus", "Oracle of Delphi", "Oracle of Dodoni", "Oracle of Delos"]
actions = ["fly", "walk", "horse"]

# Transition probabilities and rewards as provided in the MDP diagram
# P[s][a] = {'next_state': probability of transition}
# R[s][a] = {'next_state': reward received}
P = {
    "Mount Olympus": {
        "fly": {"Oracle of Delphi": 0.9, "Mount Olympus": 0.1},
        "walk": {"Oracle of Delphi": 0.2, "Oracle of Dodoni": 0.8}
    },
    "Oracle of Delphi": {
        "fly": {"Oracle of Delos": 0.7, "Oracle of Delphi": 0.3},
        "horse": {"Mount Olympus": 0.8, "Oracle of Dodoni": 0.2}
    },
    "Oracle of Dodoni": {
        "fly": {"Mount Olympus": 0.7, "Oracle of Dodoni": 0.3},
        "horse": {"Oracle of Delphi": 0.3, "Mount Olympus": 0.7}
    },
    "Oracle of Delos": {
        "fly": {"Oracle of Dodoni": 0.4, "Oracle of Delphi": 0.4, "Oracle of Delos": 0.2}
    }
}

R = {
    "Mount Olympus": {
        "fly": {"Oracle of Delphi": 2, "Mount Olympus": -1},
        "walk": {"Oracle of Delphi": -2, "Oracle of Dodoni": 2}
    },
    "Oracle of Delphi": {
        "fly": {"Oracle of Delos": 6, "Oracle of Delphi": -1},
        "horse": {"Mount Olympus": 1, "Oracle of Dodoni": 1}
    },
    "Oracle of Dodoni": {
        "fly": {"Mount Olympus": 2, "Oracle of Dodoni": -1},
        "horse": {"Oracle of Delphi": 5}
    },
    "Oracle of Delos": {
        "fly": {"Oracle of Dodoni": -1, "Oracle of Delphi": -1, "Oracle of Delos": -1}
    }
}

V = {state: 0 for state in states}
gamma = 0.9
policy = {
    "Mount Olympus": {"walk": 1.0},
    "Oracle of Delphi": {"horse": 0.7, "fly": 0.3},
    "Oracle of Dodoni": {"horse": 1.0},
    "Oracle of Delos": {"fly": 1.0}
}

def policy_evaluation(V, policy, P, R, gamma, threshold=0.01):
    while True:
        delta = 0
        for state in states:
            v = 0
            for action, action_prob in policy[state].items():
                for next_state, trans_prob in P[state][action].items():
                    reward = R[state][action].get(next_state, 0)
                    v += action_prob * trans_prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[state]))
            V[state] = v
        if delta < threshold:
            break
    return V

def policy_improvement(V, policy, P, R, gamma):
    policy_stable = True
    for state in states:
        # Create a dictionary to store the expected return for each action
        action_returns = {}
        for action in P[state]:
            action_return = 0
            for next_state, prob in P[state][action].items():
                reward = R[state][action].get(next_state, 0)
                action_return += prob * (reward + gamma * V[next_state])
            action_returns[action] = action_return
        
        # Find the best action by comparing expected returns
        best_action = max(action_returns, key=action_returns.get)

        # Update the policy to be greedy with respect to the expected returns
        for action in actions:
            policy[state][action] = 0.0
        policy[state][best_action] = 1.0

        # Check if the policy has changed significantly
        old_action = max(policy[state], key=policy[state].get)
        if old_action != best_action:
            policy_stable = False

    return policy, policy_stable


# Initialize the value function and the policy
V = {state: 0 for state in states}
policy_stable = False
iteration = 0

# Iterate policy evaluation and improvement until convergence
while not policy_stable:
    V = policy_evaluation(V, policy, P, R, gamma)
    policy, policy_stable = policy_improvement(V, policy, P, R, gamma)
    iteration += 1
    if iteration >= 10:  # Prevent infinite loops by setting a maximum number of iterations
        break

print()
print("Optimal Value Function:", V)
print()
print("Optimal Policy:", policy)
