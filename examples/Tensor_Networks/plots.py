import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle
    import matplotlib.pyplot as plt

    return pickle, plt


@app.cell
def _(pickle):
    with open("large_hchain_results.pkl", "rb") as f:
        hchain_data = pickle.load(f)
    return (hchain_data,)


@app.cell
def _(hchain_data):
    energies = hchain_data["energies"]
    times = hchain_data["times"]
    hf_energy = hchain_data["hf_energy"]
    exact_energy = hchain_data["exact_energy"]
    return energies, exact_energy, hf_energy, times


@app.cell
def _(energies, exact_energy, hf_energy, plt, times):
    fig, ax = plt.subplots()
    right_ax = ax.twinx()
    # sns.lineplot(data=df, x="iteration", y="abs_error", ax=ax)
    # sns.lineplot(data=df, x="iteration", y="time", ax=right_ax)
    ax.plot(range(len(energies)), energies, color="blue")
    right_ax.plot(range(len(times)), times, color="red")
    ax.hlines(hf_energy, 0, len(energies) -1, colors="k", label="HF")
    ax.hlines(exact_energy, 0, len(energies) - 1, colors="orange", label="FCI")
    ax.set_ylabel("energy", color="blue")
    ax.set_xlabel("Iteration")
    right_ax.set_ylabel("Iteration time", color="red")
    ax.legend()
    plt.savefig("hchain_timing.pdf")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
