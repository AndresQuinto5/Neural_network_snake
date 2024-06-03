'''

this file is for the creation of a neural network that learns to play the game snake

'''

import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices([], 'GPU')

# Definir constantes del juego
WIDTH = 800
HEIGHT = 600
BLOCK_SIZE = 20
FPS = 10

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

class Snake:
    def __init__(self):
        self.positions = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = random.choice([pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT])

    def move(self):
        head_x, head_y = self.positions[0]
        if self.direction == pygame.K_UP:
            new_head = (head_x, head_y - BLOCK_SIZE)
        elif self.direction == pygame.K_DOWN:
            new_head = (head_x, head_y + BLOCK_SIZE)
        elif self.direction == pygame.K_LEFT:
            new_head = (head_x - BLOCK_SIZE, head_y)
        elif self.direction == pygame.K_RIGHT:
            new_head = (head_x + BLOCK_SIZE, head_y)
        self.positions.insert(0, new_head)
        self.positions.pop()

    def draw(self, screen):
        for position in self.positions:
            pygame.draw.rect(screen, WHITE, (position[0], position[1], BLOCK_SIZE, BLOCK_SIZE))

class Food:
    def __init__(self):
        self.position = self.generate_position()

    def generate_position(self):
        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return (x, y)

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.position[0], self.position[1], BLOCK_SIZE, BLOCK_SIZE))

class SnakeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_state(snake, food):
    head_x, head_y = snake.positions[0]
    food_x, food_y = food.position

    state = [
        # Danger straight
        (snake.direction == pygame.K_RIGHT and head_x + BLOCK_SIZE >= WIDTH) or 
        (snake.direction == pygame.K_LEFT and head_x - BLOCK_SIZE < 0) or 
        (snake.direction == pygame.K_DOWN and head_y + BLOCK_SIZE >= HEIGHT) or 
        (snake.direction == pygame.K_UP and head_y - BLOCK_SIZE < 0),

        # Danger right
        (snake.direction == pygame.K_UP and head_x + BLOCK_SIZE >= WIDTH) or 
        (snake.direction == pygame.K_DOWN and head_x - BLOCK_SIZE < 0) or 
        (snake.direction == pygame.K_LEFT and head_y + BLOCK_SIZE >= HEIGHT) or 
        (snake.direction == pygame.K_RIGHT and head_y - BLOCK_SIZE < 0),

        # Danger left
        (snake.direction == pygame.K_DOWN and head_x + BLOCK_SIZE >= WIDTH) or 
        (snake.direction == pygame.K_UP and head_x - BLOCK_SIZE < 0) or 
        (snake.direction == pygame.K_RIGHT and head_y + BLOCK_SIZE >= HEIGHT) or 
        (snake.direction == pygame.K_LEFT and head_y - BLOCK_SIZE < 0),

        # Move direction
        snake.direction == pygame.K_LEFT,
        snake.direction == pygame.K_RIGHT,
        snake.direction == pygame.K_UP,
        snake.direction == pygame.K_DOWN,

        # Food location 
        food_x < head_x,  # Food left
        food_x > head_x,  # Food right
        food_y < head_y,  # Food up
        food_y > head_y   # Food down
    ]

    return np.array(state, dtype=int)

def game_loop():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    snake = Snake()
    food = Food()
    agent = SnakeAgent(state_size=11, action_size=4)

    batch_size = 32
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        state = get_state(snake, food)
        action = agent.act(state)

        if action == 0:
            snake.direction = pygame.K_UP
        elif action == 1:
            snake.direction = pygame.K_DOWN
        elif action == 2:
            snake.direction = pygame.K_LEFT
        elif action == 3:
            snake.direction = pygame.K_RIGHT

        snake.move()

        reward = 0
        if snake.positions[0] == food.position:
            reward = 10
            snake.positions.append(snake.positions[-1])
            food.position = food.generate_position()
        elif snake.positions[0][0] < 0 or snake.positions[0][0] >= WIDTH or snake.positions[0][1] < 0 or snake.positions[0][1] >= HEIGHT or snake.positions[0] in snake.positions[1:]:
            reward = -10
            done = True

        next_state = get_state(snake, food)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        screen.fill(BLACK)
        snake.draw(screen)
        food.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    game_loop()