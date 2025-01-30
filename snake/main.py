import multiprocessing
import numpy as np
import torch
from src.agents.snake_agent import Agent
from src.games.snake_game_ai import SnakeGameAI
from src.agents.helper import plot

NUM_PROCESSES = 4
MAX_GAMES = 1000  # Define max games to avoid memory issues

def update_plot(shared_plot_scores, shared_mean_scores):
    scores = np.ctypeslib.as_array(shared_plot_scores.get_obj()).reshape(NUM_PROCESSES, MAX_GAMES)
    mean_scores = np.ctypeslib.as_array(shared_mean_scores.get_obj()).reshape(NUM_PROCESSES, MAX_GAMES)
    plot(scores.tolist(), mean_scores.tolist())  # Convert back to normal lists for plotting

def launch_training(agent_id, shared_plot_scores, shared_mean_scores, shared_counts):
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train agent memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Agent {agent_id} - Game {agent.n_games} - Score {score} - Record: {record}')

            # Save data to shared memory
            idx = shared_counts[agent_id]
            if idx < MAX_GAMES:  # Prevent out-of-bounds access
                shared_plot_scores[agent_id * MAX_GAMES + idx] = score
                total_score += score
                mean_score = total_score / agent.n_games
                shared_mean_scores[agent_id * MAX_GAMES + idx] = mean_score
                shared_counts[agent_id] += 1  # Increment count


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    # Use shared memory arrays instead of Manager()
    shared_plot_scores = multiprocessing.Array('d', NUM_PROCESSES * MAX_GAMES)  # Shared array for scores
    shared_mean_scores = multiprocessing.Array('d', NUM_PROCESSES * MAX_GAMES)  # Shared array for mean scores
    shared_counts = multiprocessing.Array('i', [0] * NUM_PROCESSES)  # Track index per agent

    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=launch_training, args=(i, shared_plot_scores, shared_mean_scores, shared_counts))
        p.start()
        processes.append(p)
    
    while True:
        update_plot(shared_plot_scores, shared_mean_scores)

    for p in processes:
        p.join()
