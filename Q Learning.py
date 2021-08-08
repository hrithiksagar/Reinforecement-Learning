import gym
print("imported Gym")

env = gym.make("MountainCar-v0")
env.reset()

done = False
while not done:
    action = 2 # 0 = push car left 1= nothing and actioon 2 = push car right
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()