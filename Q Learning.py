# Goal is to reach the flag
import gym
print("imported Gym")

env = gym.make("MountainCar-v0")
env.reset()
print(env.observation_space.low)
print(env.observation_space.low)
print(env.action_space.n)
DISCRETE_OS_SIZE = [20]*len(env.observation_space.low)
discrete_os_win_size = (env.observation_space.low - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)
done = False
while not done:
    action = 2 # 0 = push car left 1= nothing  and action 2 = push car right
    new_state, reward, done, _ = env.step(action)
    #print(new_state)
    #print(reward)
    env.render()

env.close()
