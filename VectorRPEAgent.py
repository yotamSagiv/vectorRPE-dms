import numpy as np


class VectorRPEAgent(object):
    def __init__(self, num_features, lr, gamma):
        self.num_features = num_features
        self.alpha = lr
        self.gamma = gamma
        self.weights = np.zeros(num_features)

    def val(self, state_vec):
        return np.dot(state_vec, self.weights)

    def compute_delta_feat(self, state_vec, succ_vec, reward):
        delta_features = np.zeros(self.num_features)
        for i in range(self.num_features):
            delta_features[i] = reward / self.num_features
            delta_features[i] += self.weights[i] * (self.gamma * succ_vec[i] - state_vec[i])

        return delta_features

    def compute_delta(self, state_vec, succ_vec, reward):
        return np.sum(self.compute_delta_feat(state_vec, succ_vec, reward))

    def learn(self, state_vec, succ_vec, reward, ret_da=True):
        delta_feat = self.compute_delta_feat(state_vec, succ_vec, reward)
        delta = np.sum(delta_feat)
        self.weights += self.alpha * delta * state_vec

        if ret_da:
            return delta_feat


def phi_tabular(state, max_state):
    if state > max_state:
        raise Exception('For this simple example, states can only be integers between 0 and max_state')

    state_vec = np.zeros(max_state + 1)
    state_vec[state] = 1

    return state_vec


def phi_poly(state, num_features):
    state_vec = np.zeros(num_features)
    for i in range(num_features):
        state_vec[i] = state ** i

    return state_vec


if __name__ == '__main__':
    # Tabular example
    max_state = 5
    lr = 0.05
    gamma = 0.95
    vrpe = VectorRPEAgent(max_state + 1, lr, gamma)

    num_trials = 10000
    rewards = [0 for i in range(max_state)]
    rewards.append(1)

    curr_state = 0
    for t in range(num_trials):
        next_state = curr_state + 1
        vrpe.learn(phi_tabular(curr_state, max_state), phi_tabular(next_state, max_state), rewards[next_state])

        if next_state == max_state:
            curr_state = 0
        else:
            curr_state = next_state

    print('Tabular weights:', vrpe.weights)

    # Polynomial basis example
    max_state = 5
    num_features = 3
    lr = 0.3
    gamma = 0.95
    vrpe = VectorRPEAgent(num_features, lr, gamma)

    num_trials = 10000
    rewards = [0 for i in range(max_state)]
    rewards.append(1)

    curr_state = 0
    for t in range(num_trials):
        if t == 4:
            pass
        next_state = curr_state + 1
        vrpe.learn(phi_poly(curr_state / max_state, num_features), phi_poly(next_state / max_state, num_features),
                   rewards[next_state])
        print(t)
        print(vrpe.weights)

        if next_state == max_state:
            curr_state = 0
        else:
            curr_state = next_state

    print('Polynomial weights:', vrpe.weights)
    print('Polynomial values:')

    for i in range(max_state + 1):
        print(vrpe.val(phi_poly(i / max_state, num_features)))







