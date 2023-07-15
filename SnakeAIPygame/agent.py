import torch
import numpy as np
import random
from collections import deque

from settings import Settings
from game import SnakeGameAI, Direction, Point
from model import LinearQNet, QTrainer
from plot import plot


class Agent:
    """
    A class for the AI's functionality such as its moves and its training.
    """
    def __init__(self):
        self.settings = Settings()
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=self.settings.MAX_MEMORY) # popleft()
        self.model = LinearQNet(11, 256, 3)   # for starting training from scratch
        # self.model.load_state_dict(torch.load('./model_51/model.pth'))    # for starting training from loaded model
        self.model.train()
        self.trainer = QTrainer(self.model, lr=self.settings.lr, gamma=self.gamma)

    def get_state(self, game):
        """
        Returns a list of 11 booleans as the first layer of the neural network.
        3 are whether there is danger ahead, right, or left.
        4 is the direction of the movement of the snake.
        4 are if the food is up, right, down, or left.
        """
        # points surrounding the snake head
        head = game.snake[0]
        point_l = Point(head.x - self.settings.BLOCK_SIZE, head.y)
        point_r = Point(head.x + self.settings.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - self.settings.BLOCK_SIZE)
        point_d = Point(head.x, head.y + self.settings.BLOCK_SIZE)

        # boolean directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # state returns a list of 11 booleans, the input to the neural network
        state = [
            # the first 3 booleans are whether there is danger ahead, right, or left e.g. [0,0,0,...]
            # danger straight (first boolean)
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right (second boolean)
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # danger left (third boolean)
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # the next 4 booleans are the direction of movement e.g. [...0, 1, 0, 0,...]
            dir_l, dir_r, dir_u, dir_d,

            # the last 4 booleans is the food location e.g. [...0, 1, 1, 0]
            game.food.x < game.head.x,      # food is left
            game.food.x > game.head.x,      # food is right
            game.food.y < game.head.y,      # food is up
            game.food.y > game.head.y,       # food is down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.settings.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.settings.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        :return: list of 3 booleans indicating the change in direction of snake. (no change, right, left)
        """
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        # the move will be more random early on
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)     # move a random direction
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)     # returns a list of 3 booleans (for next direction)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score

            if score > 50:
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
