import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle as pkl
    import numpy as np
    import matplotlib.pyplot as plt

    return pkl, plt


@app.cell
def _(pkl):
    with open("hchain_results_N4.pkl", "rb") as f:
        data = pkl.load(f)
    print(data.keys())
    return (data,)


@app.cell
def _(data, plt):
    fig, ax = plt.subplots()
    ax.plot(data["energies"], label="ADAPT")
    ax.hlines(data["hf_energy"], 0, len(data["energies"]) - 1, label="HF", colors="k")
    ax.hlines(data["exact_energy"], 0, len(data["energies"]) - 1, label="CCSD", colors="orange")
    # ax.hlines(data["dmrg_energy"], 0, len(data["energies"]) - 1, label="DMRG", colors="purple")
    ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    plt.savefig("hchain_energies.pdf")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
