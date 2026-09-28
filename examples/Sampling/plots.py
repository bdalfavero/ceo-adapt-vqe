import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pickle as pkl
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, pkl, plt


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
def _(energies):
    print(energies)
    return


@app.cell
def _(energies, hardware_energies, linalg_energies, plt):
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(linalg_energies)), linalg_energies, '.', color="blue", label="LinAlgAdapt")
    ax2.plot(range(len(linalg_energies), len(energies)), hardware_energies, '.', color="red", label="SampledLinAlgAdapt")
    ax2.legend()
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy")
    return


@app.cell
def _(k):
    k
    return


if __name__ == "__main__":
    app.run()
