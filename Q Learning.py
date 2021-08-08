# Goal is to reach the flag
# making a agent "Car getting up a mountain"

import gym
import numpy as np

print("imported Gym")

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.low)
print(env.observation_space.low)
print(env.action_space.n)

LEARNING_RATE = 0.1  # this is anything form 0 to 1, 0.1 is  low but we can decay rate, for now this is 0.1
DISCOUNT = 0.95  # (weight) measure of how imp do we find feature actions (how mcuh we value future)
# maxq value is imp
EPISODES = 25000
SHOW_EVERY = 2000

DISCRETE_OS_SIZE = [1000000] * len(env.observation_space.low)
discrete_os_win_size = (env.observation_space.low - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# q_table is every possible observational combination and every combination of every possible action ant every
# possible =combination we have random starting q_value

print(q_table.shape)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    if episode %  SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    # print(discrete_state)
    # print(np.argmax(q_table[discrete_state]))

    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])  # 0 = push car left 1= nothing  and action 2 = push car right
        new_state, reward, done, _ = env.step(action)
        # print(reward, new_state) # rewards are always -1 until good things happen
        new_discrete_state = get_discrete_state(new_state)  # cause we use this at formulation of new values
        env.render()
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                    reward + DISCOUNT * max_future_q)  # formula for calculating all q values
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
        discrete_state = new_discrete_state
env.close()
print("Done")
