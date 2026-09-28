import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt

    return np, pickle, plt


@app.cell
def _(np, pickle):
    with open("exact_results.pkl", "rb") as f:
        exact_data = pickle.load(f)
    exact_energies = exact_data["energies"]
    noisy_sim_energies = np.load("noisy_energies.npy")
    return exact_energies, noisy_sim_energies


@app.cell
def _(exact_energies, noisy_sim_energies, plt):
    fig, ax = plt.subplots()
    ax.plot(range(len(exact_energies)), exact_energies, '.', label="Exact Aer simulator")
    ax.plot(range(len(noisy_sim_energies)), noisy_sim_energies, '.', label="Exact training, noisy sim")
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
