import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle as pkl
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, np, pkl, plt


@app.cell
def _(pkl):
    with open("hchain_results.pkl", "rb") as f:
        data = pkl.load(f)
    return (data,)


@app.cell
def _(data):
    print(data.keys())
    return


@app.cell
def _(data, plt):
    fig, ax = plt.subplots()
    ax.plot(data["energies"], color="blue", label="ADAPT")
    ax.hlines(data["exact_energy"], 0, len(data["energies"]) - 1, colors="k", label="FCI")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    plt.savefig("hchain_hardware_energies.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handoff between `LinAlgAdapt` and `SampledLinAlgAdapt`
    """)
    return


@app.cell
def _(pkl):
    with open("xxz_handoff_results.pkl", "rb") as f:
        handoff_data = pkl.load(f)
    return (handoff_data,)


@app.cell
def _(handoff_data):
    num_linalg_iter = handoff_data["num_linalg_iter"]
    energies = handoff_data["energies"]
    linalg_energies = handoff_data["energies"][:num_linalg_iter]
    hardware_energies = handoff_data["energies"][num_linalg_iter:]
    return energies, hardware_energies, linalg_energies


@app.cell
def _(np):
    exact_energies = np.load("exact_energies.npy")
    noisy_energies = np.load("noisy_energies.npy")
    return exact_energies, noisy_energies


@app.cell
def _(
    energies,
    exact_energies,
    hardware_energies,
    linalg_energies,
    noisy_energies,
    plt,
):
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(linalg_energies)), linalg_energies, '.', color="blue", label="LinAlgAdapt")
    ax2.plot(range(len(linalg_energies), len(energies)), hardware_energies, '.', color="red", label="SampledLinAlgAdapt")
    ax2.plot(range(len(exact_energies)), exact_energies, "v", color="orange", alpha=0.5, label="Exact simulator")
    ax2.plot(range(len(noisy_energies)), noisy_energies, ">", color="purple", alpha=0.5, label="Noisy simulator")
    ax2.legend()
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy")
    plt.savefig("sampling.pdf")
    return


if __name__ == "__main__":
    app.run()
