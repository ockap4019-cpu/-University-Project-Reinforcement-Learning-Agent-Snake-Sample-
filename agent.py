import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Silencia el warning de pkg_resources (opcional)

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point   # baseline: solo SnakeGameAI
from model import Linear_QNet, QTrainer
from helper import plot, save_results
import matplotlib.pyplot as plt

# Semillas para reproducibilidad (opcional)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Parámetros baseline (Parte A)
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        # baseline NN: 2 capas ocultas (p.ej., 256 y 256)
        self.model = Linear_QNet(11, [256, 256], 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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

        state = [
            # peligro al frente
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # peligro a la derecha
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # peligro a la izquierda
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # dirección de movimiento
            dir_l, dir_r, dir_u, dir_d,

            # ubicación de la comida
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = list(self.memory)

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # epsilon-greedy con clamp para evitar valores negativos
        self.epsilon = max(0, 80 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train_baseline():
    print("🚀 Starting training loop (Part A, 500 episodes)...")

    scores = []
    mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while len(scores) < 500:  # Parte A: 500 episodios
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            mean_scores.append(mean_score)
            if score > record:
                record = score

            # Traza de progreso
            print(f"Game {agent.n_games} | Score: {score} | Mean: {mean_score:.2f} | Record: {record}")

            # Actualiza gráfico interactivo (si tu helper.plot lo soporta)
            plot(scores, mean_scores)

    # Guardar resultados al finalizar
    save_results(scores, mean_scores, filename="baseline_results.csv")

    plt.figure()
    plt.title("Training Results (Baseline NN)")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.legend()
    plt.savefig("training_plot_baseline.png")

    print("🏁 Training completed: 500 episodes")
    print("📄 CSV saved as 'baseline_results.csv'")
    print("🖼️ Plot saved as 'training_plot_baseline.png'")


if __name__ == '__main__':
    train_baseline()

