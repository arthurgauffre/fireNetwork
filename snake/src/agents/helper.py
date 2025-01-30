import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(shared_plot_scores, shared_mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    for i in range(len(shared_plot_scores)):
        plt.plot(shared_plot_scores[i])
    for i in range(len(shared_mean_scores)):
        plt.plot(shared_mean_scores[i], color='red')
    plt.ylim(ymin=0)
    for i in range(len(shared_plot_scores)):
        plt.text(len(shared_plot_scores[i])-1, shared_plot_scores[i][-1], str(shared_plot_scores[i][-1]))
    for i in range(len(shared_mean_scores)):
        plt.text(len(shared_mean_scores[i])-1, shared_mean_scores[i][-1], str(shared_mean_scores[i][-1]))
    plt.show(block=False)
    plt.pause(.1)