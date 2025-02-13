import torch
import random
import numpy as np
from collections import deque
from src.games.snake_game_ai import Direction, Point
from src.agents.model import Linear_QNet, QTrainer
from src.agents.helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(59, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.max_score = 0


    def get_state(self, game):
        head = game.snake[0]
        
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        grid_size = 5
        grid_state = []
        for dy in [-2, -1, 0, 1, 2]:
            for dx in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    continue
                cell_x = head.x + dx * BLOCK_SIZE
                cell_y = head.y + dy * BLOCK_SIZE
                point = Point(cell_x, cell_y)
                
                boundary = (cell_x < 0 or cell_x >= game.w or 
                            cell_y < 0 or cell_y >= game.h)
                
                body = point in game.snake[1:]
                
                food = point == game.food
                
                grid_state.extend([
                    boundary or body,
                    food
                ])

        basic_state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        full_state = np.array(basic_state + grid_state, dtype=int)
        return full_state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_agent(self, file_name='agent.pth', folder_path='./model'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,
            'max_score': self.max_score
        }, file_name)

    def load_agent(self, file_name='agent.pth'):
        file_name = os.path.join('./model', file_name)
        if os.path.exists(file_name):
            checkpoint = torch.load(file_name, map_location=torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state'])
            self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.n_games = checkpoint['n_games']
            self.max_score = checkpoint['max_score']
            print(f"Loaded agent state from {file_name}")
        else:
            print(f"No saved state found at {file_name}")
