from network import Network
import gym

if __name__ == "__main__":
    #start the test environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    n_inputs = env.observation_space.shape[0]
    n_outputs = env.action_space.shape[0]
    net = Network(f"Networks/{input('Select Network: ')}", n_inputs, n_outputs, Network.ActivationFunction.Tanh)

    #test the networks on the environment
    total_reward = 0
    observation, reward, done, info = (0, 0), 0, False, {}
    for t in range(1000):
        env.render()
        act = net.CalculateOutputs(observation)
        observation, reward, done, info = env.step(act) #do an action
        total_reward += reward #store the reward for this iteration
