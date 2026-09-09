import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle as pkl
    import matplotlib.pyplot as plt

    return pkl, plt


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
