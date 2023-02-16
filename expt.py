import numpy as np
import matplotlib.pyplot as plt

from VectorRPEAgent import VectorRPEAgent

np.random.seed(865612)

def build_phi_simple(press_thresh):
    def phi_simple(press_count):
        vec = np.zeros(press_thresh + 1)
        vec[press_count] = 1

        return vec

    return phi_simple


def build_phi_twohot(press_thresh):
    def phi_twohot(press_counts):
        lvec = np.zeros(press_thresh + 1)
        lvec[press_counts[0]] = 1

        rvec = np.zeros(press_thresh + 1)
        rvec[press_counts[1]] = 1

        return np.hstack((lvec, rvec))

    return phi_twohot


def build_phi_chunked(press_thresh, num_copies, is_weighted):
    def phi_chunked(press_count):
        phi_base = np.zeros((1, press_thresh + 1))
        phi_base[0, press_count] = 1

        if not is_weighted:
            phi = np.tile(phi_base, (num_copies, 1))
        else:
            phi = np.zeros((num_copies, press_thresh + 1))
            for i in range(num_copies):
                if i == 0:
                    phi[i, :] = phi_base
                else:
                    phi[i, :] = phi_base[0, 0]  # Should this be phi_base[1]?

        return phi.flatten()
    return phi_chunked


def softmax(vals, beta):
    return np.exp(vals / beta) / np.sum(np.exp(vals / beta))


# Bandit parameters
num_bandits = 2
press_counts = np.zeros(num_bandits, dtype=int)
bandit_rewards = np.array([10, 0])
press_thresh = 8

# Task parameters
num_timesteps = 10000
num_trials = int(num_timesteps / press_thresh)  # upper bound
num_switches = 100
mean_switch = 60
scale_switch = 2
swap_trial_diffs = np.random.gamma(mean_switch, scale_switch, size=num_switches)
swap_trials = np.trunc(np.cumsum(swap_trial_diffs))
ITI_len = 5

# Agent parameters
lr = 0.05
gamma = 0.95
sftmx_temp = 0.5

simple_features = press_thresh + 1
twohot_features = (press_thresh + 1) * 2
num_copies = 2
chunked_features = (press_thresh + 1) * num_copies

sl_ag = VectorRPEAgent(simple_features, lr, gamma)  # simple left agent
sr_ag = VectorRPEAgent(simple_features, lr, gamma)  # simple right agent
thl_ag = VectorRPEAgent(twohot_features, lr, gamma)  # two-hot left agent
thr_ag = VectorRPEAgent(twohot_features, lr, gamma)  # two-hot right agent
chl_ag = VectorRPEAgent(chunked_features, lr, gamma)  # chunked left agent
chr_ag = VectorRPEAgent(chunked_features, lr, gamma)  # chunked left agent
phi_simple = build_phi_simple(press_thresh)
phi_twohot = build_phi_twohot(press_thresh)
phi_chunked = build_phi_chunked(press_thresh, num_copies, True)

da_simple_L = np.zeros((num_timesteps, simple_features))
da_simple_R = np.zeros((num_timesteps, simple_features))
da_twohot_L = np.zeros((num_timesteps, twohot_features))
da_twohot_R = np.zeros((num_timesteps, twohot_features))
da_chunk_L = np.zeros((num_timesteps, chunked_features))
da_chunk_R = np.zeros((num_timesteps, chunked_features))
actions = np.zeros(num_timesteps) + np.nan  # nan not a valid action
trial_times = np.zeros(num_trials + 1) - 1  # -1 obvious if something has gone wrong

l_weights = np.zeros((num_timesteps, chunked_features)) + np.nan
# Simulate
tdx = 0
ITI_timesteps = 0
# swap_trials = []
for t in range(num_timesteps):
    if t in swap_trials:  # Check for reversal
        bandit_rewards = bandit_rewards[::-1]
    if not np.any(press_counts) and ITI_timesteps == 0:
        trial_times[tdx] = t
        tdx += 1
    if ITI_timesteps > 0:
        ITI_timesteps -= 1
        continue
    l_weights[t] = chl_ag.weights

    # Extract current state
    left_simple_state = phi_simple(press_counts[0])
    right_simple_state = phi_simple(press_counts[1])
    left_chunked_state = phi_chunked(press_counts[0])
    right_chunked_state = phi_chunked(press_counts[1])
    twohot_state = phi_twohot(press_counts)

    # Pick action (assume agent knows the reward values)
    action = np.random.choice(2, p=softmax(bandit_rewards, sftmx_temp))
    actions[t] = action

    # Update state
    press_counts[action] += 1
    left_simple_succ_state = phi_simple(press_counts[0])
    right_simple_succ_state = phi_simple(press_counts[1])
    left_chunked_succ_state = phi_chunked(press_counts[0])
    right_chunked_succ_state = phi_chunked(press_counts[1])
    twohot_succ_state = phi_twohot(press_counts)

    # Learn
    l_reward = int(action == 0)
    r_reward = int(action == 1)

    da_simple_L[t, :] = sl_ag.learn(left_simple_state, left_simple_succ_state, l_reward)
    da_simple_R[t, :] = sr_ag.learn(right_simple_state, right_simple_succ_state, r_reward)
    da_chunk_L[t, :] = chl_ag.learn(left_chunked_state, left_chunked_succ_state, l_reward)
    da_chunk_R[t, :] = chr_ag.learn(right_chunked_state, right_chunked_succ_state, r_reward)
    da_twohot_L[t, :] = thl_ag.learn(twohot_state, twohot_succ_state, l_reward)
    da_twohot_R[t, :] = thr_ag.learn(twohot_state, twohot_succ_state, r_reward)

    if np.any(da_chunk_L[t, :] > 100):
        print(t)
    if np.any(chl_ag.weights > 30):
        print(t)

    if press_thresh in press_counts:
        ITI_timesteps = ITI_len
        press_counts[:] = 0

np.savez('DA_trace_lowtemp.npz', simple_L=da_simple_L, simple_R=da_simple_R, twohot_L=da_twohot_L, twohot_R=da_twohot_R,
         reward_mags=bandit_rewards, temp=sftmx_temp, actions=actions, trial_times=trial_times, chunk_L=da_chunk_L,
         chunk_R=da_chunk_R, num_copies=num_copies)












