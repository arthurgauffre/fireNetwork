from src.agents.snake_agent import Agent
from src.games.snake_game_ai import SnakeGameAI
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Manager
import threading
import torch

NUM_PROCESSES = 4

def launch_training(agent_id, file_lock, shared_scores, plot_lock):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(file_lock)  # Pass the lock to Agent
    game = SnakeGameAI(display=(agent_id == 0))

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

            with plot_lock:
                shared_scores.append(score)
            
            print(f'Agent {agent_id} - Game {agent.n_games} - Score {score} - Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

def plot_updater(shared_scores, plot_lock):
    all_scores = []
    plt.ion()  # Enable interactive mode
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    while True:
        with plot_lock:
            new_scores = list(shared_scores)
        
        if new_scores:
            all_scores.extend(new_scores)
            shared_scores[:] = []  # Clear shared list
            
            mean_scores = []
            total_score = 0
            for i, score in enumerate(all_scores, 1):
                total_score += score
                mean_scores.append(total_score / i)
            
            # Update plot
            ax.clear()
            ax.set_title('Training...')
            ax.set_xlabel('Number of Games')
            ax.set_ylabel('Score')
            ax.plot(all_scores)
            ax.plot(mean_scores, color='red')
            ax.set_ylim(ymin=0)
            if all_scores:
                ax.text(len(all_scores)-1, all_scores[-1], str(all_scores[-1]))
            if mean_scores:
                ax.text(len(mean_scores)-1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
            plt.show(block=False)
            plt.pause(0.1)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    manager = Manager()
    shared_scores = manager.list()
    file_lock = multiprocessing.Lock()
    plot_lock = multiprocessing.Lock()

    # Start plot updater thread
    plot_thread = threading.Thread(
        target=plot_updater,
        args=(shared_scores, plot_lock),
        daemon=True
    )
    plot_thread.start()

    # Start training processes
    processes = []
    for i in range(NUM_PROCESSES):
        p = multiprocessing.Process(
            target=launch_training,
            args=(i, file_lock, shared_scores, plot_lock)
        )
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        for p in processes:
            p.terminate()