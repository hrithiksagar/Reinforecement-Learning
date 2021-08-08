# Goal is to reach the flag
import gym
import numpy as np
print("imported Gym")

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.low)
print(env.observation_space.low)
print(env.action_space.n)
DISCRETE_OS_SIZE = [20] * len(env.observation_space.low)
discrete_os_win_size = (env.observation_space.low - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE+[env.action_space.n]))
# q_table is every possible observational combination and every combination of every possible action ant every
# possible =combination we have random starting q_value

print(q_table.shape)

done = False
while not done:
    action = 2  # 0 = push car left 1= nothing  and action 2 = push car right
    new_state, reward, done, _ = env.step(action)
    #print(reward, new_state) # rewards are always -1 until good things happen

    env.render()
env.close()
