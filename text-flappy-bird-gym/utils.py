import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap_from_Q(Q, ax, cmap='viridis'):
    X, Y, Z = [], [], []

    for state, q_values in Q.items():
        if not isinstance(state, tuple):
            continue

        x, y = state
        X.append(x)
        Y.append(y)
        Z.append(np.max(q_values))

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    contour = ax.tricontourf(X, Y, Z, levels=20, cmap=cmap)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("V(x,y)")

    return contour

def plot_score(n_runs, env, agent_mc, agent_sarsa):
    scores_mc = agent_mc.test(env, n_runs = n_runs)
    scores_sarsa = agent_sarsa.test(env, n_runs = n_runs)

    mean_mc = np.mean(scores_mc)
    std_mc = np.std(scores_mc)
    mean_sarsa = np.mean(scores_sarsa)
    std_sarsa = np.std(scores_sarsa)

    print(f"Mean score for MC : {mean_mc}")
    print(f"Mean score for SARSA : {mean_sarsa}")

    print(f"STD for MC : {std_mc}")
    print(f"STD for SARSA : {std_sarsa}")

    plt.figure(figsize=(10, 5))

    # Histogrammes superposés
    plt.hist(scores_mc, bins=20, alpha=0.5, label="Monte Carlo")
    plt.hist(scores_sarsa, bins=20, alpha=0.5, label="SARSA")

    # Lignes de moyenne
    plt.axvline(mean_mc, linestyle='--', label=f"MC mean = {mean_mc:.2f}", color = "red")
    plt.axvline(mean_sarsa, linestyle='--', label=f"SARSA mean = {mean_sarsa:.2f}", color = "blue")

    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("Score Distribution (MC vs SARSA)")
    plt.legend()

    plt.show()


def compare_hyperparameters(env, agents, num_episodes, str_parameter, range_parameter, n_runs = 1_000):
    for agent in agents:
        les_x = range_parameter
        les_y = []

        for parameter in range_parameter:
            setattr(agent, str_parameter, parameter)
            agent.reset()
            agent.train(env, num_episodes=num_episodes)
            scores = agent.test(env, n_runs = n_runs)
            les_y.append(np.mean(scores))

        plt.plot(les_x, les_y, label = f"Average score for {agent.name}")
    plt.xlabel(str_parameter)
    plt.ylabel(f"Average score ({n_runs} runs)")
    plt.title(f"Average score with respect to {str_parameter}")
    plt.legend()
    plt.show()











