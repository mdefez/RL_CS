import os, sys
import gymnasium as gym
import time
from agents import MonteCarloControlAgent, SarsaLambdaAgent, ValueVisualizer
from text_flappy_bird_gym.envs.text_flappy_bird_env_simple import TextFlappyBirdEnvSimple

if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs = env.reset()[0]

    choose_agent = "MC"     # "SARSA", "MC"

    if choose_agent == "MC":
        agent = MonteCarloControlAgent(action_space_shape = env.action_space.n, gamma=0.99, epsilon=0.2)
        agent.train(env = env, num_episodes = 10_000)
        

    elif choose_agent == "SARSA":
        agent = SarsaLambdaAgent(action_space_shape = env.action_space.n, 
                                       gamma=0.99, alpha=0.1, epsilon=0.3, lam=0.9)
        agent.train(env = env, num_episodes = 10_000)
        

    obs = env.reset()[0]
    agent.test()
    
    # iterate
    while True:
        
        # Select next action
        action = agent.policy(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        os.system("cls")
        sys.stdout.write(env.render())
        time.sleep(0.02) # FPS

        # If player is dead break
        if done:
            break

    env.close()
