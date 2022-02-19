# https://ai.stackexchange.com/questions/34570/are-vs-and-pis-initialization-really-arbitrarily-in-policy-iteration

states = ['s1', 's2', 's3']
actions = {}

gamma = 1.0
values = {'s1': 10, 's2': 10, 's3': 0}
policy = {'s1': 'a1', 's2': 'a3', 's3': 'a5'}

actions['s1'] = {
    'a1': [0, 's2'],
    'a2': [1, 's3']
}

actions['s2'] = {
    'a3': [0, 's1'],
    'a4': [1, 's3']
}

actions['s3'] = {
    'a5': [0, 's3']
}

def evaluate():
    global values
    delta = 0

    for state in states:
        previous_value = values[state]
        reward, target_state = actions[state][policy[state]]
        values[state] = reward + gamma * values[target_state]
        delta = max(delta, abs(previous_value - values[state]))

    return delta

def update_policy():
    global policy
    changed = False

    for state in states:
        action_values = {}
        for action in actions[state]:
            reward, target_state = actions[state][action]
            action_values[action] = reward + gamma * values[target_state]
        best_action = max(action_values, key=action_values.get)
        if policy[state] != best_action:
            policy[state] = best_action
            changed = True

    return changed              

changed = True
while changed:
    changed = False
    # Evaluate the policy
    while evaluate() > 1e-6:
        pass

    # Update the policy
    changed = update_policy()

for state in states:
    print('State: %s, policy action: %s, value: %f' % (state, policy[state], values[state]))

optimal_policy = {'s1': 'a2', 's2': 'a4', 's3': 'a5'}
if policy == optimal_policy:
    print('This is the optimal  policy')
else:
    print('This is not the optimal policy')