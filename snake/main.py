from src.agents.snake_agent import Agent
from src.games.snake_game_ai import SnakeGameAI
from src.agents.helper import plot
import multiprocessing
import torch

NUM_PROCESSES = 4

def launch_training(agent_id, file_lock):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(file_lock)  # Pass the lock to Agent
    game = SnakeGameAI()

    agent.load_agent()
    record = agent.max_score
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
            agent.check_agent()
            record = agent.max_score
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.max_score = record  # Update agent's max_score
                agent.model.save()
                agent.save_agent()

            print(f'Agent {agent_id} - Game {agent.n_games} - Score {score} - Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    
    # Create a lock for file operations
    file_lock = multiprocessing.Lock()

    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(target=launch_training, args=(i, file_lock))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()