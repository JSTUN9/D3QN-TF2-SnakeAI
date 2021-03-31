from model import Agent
import numpy as np
from gamee import SnakeGameAI
from utils import plotLearning

if __name__ == '__main__':
    game = SnakeGameAI()
    n_games = 400
    agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[11], 
                  epsilon_dec=1e-3, mem_size=100000, batch_size=64, eps_end=0.01,
                  fc1_dims=128, fc2_dims=128, replace=100, n_actions=3)

    scores, eps_history = [], []

    for i in range(n_games):
        done = False
        score = 0
        game.reset()
        while not done:
            state = game.get_state()
            action = agent.choose_action(state)
            reward, done, score = game.play_step(action)
            state_ = game.get_state()
            score += reward
            agent.store_transition(state, action, reward, state_, done)
            
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    filename='snakeAI_Results.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)

