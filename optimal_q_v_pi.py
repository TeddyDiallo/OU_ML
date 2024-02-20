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
        "horse": {"Oracle of Delphi": 5, "Mount Olympus": 2}
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
# Initialize Q values
Q = {s: {a: 0 for a in actions if a in P[s]} for s in states}


# Value Iteration algorithm to find V* and Q*
def value_iteration(states, actions, P, R, V, Q, gamma, threshold=1e-12):
    while True:
        delta = 0
        for s in states:
            V_temp = V[s]
            for a in P[s]:
                Q[s][a] = sum([P[s][a][next_state] * (R[s][a].get(next_state, 0) + gamma * V[next_state]) for next_state in P[s][a]])
            V[s] = max(Q[s].values())
            delta = max(delta, abs(V_temp - V[s]))
        if delta < threshold:
            break


# Extract the optimal policy from Q
def extract_policy(Q):
    policy_optimal = {}
    for s in Q:
        policy_optimal[s] = max(Q[s], key=Q[s].get)
    return policy_optimal

# Run Value Iteration
value_iteration(states, actions, P, R, V, Q, gamma)

# Extract the optimal policy
pi_star = extract_policy(Q)

print("Optimal Q values:")
for s in Q:
    print(s, Q[s])

print("\nOptimal V values:")
for s in V:
    print(s, V[s])

print("\nOptimal policy:")
for s in pi_star:
    print(s, pi_star[s])
