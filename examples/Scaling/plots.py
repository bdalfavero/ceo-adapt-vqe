import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    return np, os, pd, plt, sns


@app.cell
def _(pd):
    df = pd.read_csv("hchain_results.csv")
    print(df.head())
    return (df,)


@app.cell
def _(df, plt):
    fig, ax = plt.subplots()
    right_ax = ax.twinx()
    # sns.lineplot(data=df, x="iteration", y="abs_error", ax=ax)
    # sns.lineplot(data=df, x="iteration", y="time", ax=right_ax)
    ax.plot(df["iteration"], df["abs_error"], color="blue")
    right_ax.plot(df["iteration"], df["time"], color="red")
    ax.set_ylabel("Aboslute error", color="blue")
    ax.set_xlabel("Iteration")
    right_ax.set_ylabel("Iteration time", color="red")
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XXZ
    """)
    return


@app.cell
def _(os, pd):
    data_dir = "xxz_new_new"
    data_files = os.listdir(data_dir)
    xxz_dfs = []
    for f in data_files:
        xxz_dfs.append(pd.read_csv(data_dir + "/" + f))
    df_xxz = pd.concat(xxz_dfs)

    # df_xxz = pd.read_csv("xxz_results.csv")
    print(df_xxz.head())
    return (df_xxz,)


@app.cell
def _(df_xxz, np):
    l_vals = np.unique(df_xxz["N"])
    chi_vals = np.unique(df_xxz["chi"])
    print(l_vals, chi_vals)
    return chi_vals, l_vals


@app.cell
def _(chi_vals, df_xxz, l_vals, np, plt):
    fig_xxz, ax_xxz = plt.subplots(len(l_vals), len(chi_vals), figsize=(len(l_vals) * 4.0, len(chi_vals) * 2.0))

    for i, l in enumerate(np.unique(df_xxz["N"])):
        for j, chi in enumerate(np.unique(df_xxz["chi"])):
            ax_xxz[i, j].set_title(f"L={l} chi={chi}")
            df_queried = df_xxz.query(f"N=={l} and chi=={chi}")
            right_ax_xxz = ax_xxz[i,j].twinx()
            ax_xxz[i,j].plot(df_queried["iteration"], df_queried["rel_error"], color="blue")
            ax_xxz[i,j].set_yscale("log")
            # right_ax_xxz.set_yscale("log")
            right_ax_xxz.plot(df_queried["iteration"], df_queried["time"], color="red")
            ax_xxz[i,j].set_ylabel("Relative error", color="blue")
            ax_xxz[i,j].set_xlabel("Iteration")
            right_ax_xxz.set_ylabel("Iteration time", color="red")

    fig_xxz.tight_layout()
    # plt.show()
    plt.savefig("xxz_scaling_new.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## comp to circuit
    """)
    return


@app.cell
def _(pd):
    df_mps = pd.read_csv("mps_to_circuit_results.csv")
    print(df_mps.head())
    return (df_mps,)


@app.cell
def _(df_mps, plt):
    fig_mps, ax_mps = plt.subplots()
    ax_right_mps = ax_mps.twinx()
    ax_mps.plot(df_mps["chi"], df_mps["error"], color="blue")
    ax_right_mps.plot(df_mps["chi"], df_mps["depths"], color="red")
    ax_mps.set_yscale("log")
    ax_mps.set_ylabel("Absolute Energy Error", color="blue")
    ax_right_mps.set_ylabel("Circuit Depth", color="red")
    # plt.show()
    plt.savefig("hchain_mps_circuits_N8.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison with MPS to circuit
    """)
    return


@app.cell
def _(pd):
    df_comp = pd.read_csv("hchain_adapt_bond_results.csv")
    print(df_comp.head())
    return (df_comp,)


@app.cell
def _(df_comp, plt):
    fig_comp, ax_comp  = plt.subplots()
    ax_right_comp = ax_comp.twinx()
    ax_comp.plot(df_comp["chi"], df_comp["error"], color="blue")
    ax_right_comp.plot(df_comp["chi"], df_comp["depths"], color="red")
    ax_comp.set_yscale("log")
    ax_comp.set_ylabel("Absolute Energy Error", color="blue")
    ax_right_comp.set_ylabel("Circuit Depth", color="red")
    # plt.show()
    plt.savefig("hchain_adapt_circuits_N8.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Larger scaling study
    """)
    return


@app.cell
def _(pd):
    datafiles = [
        "xxz_bond/xxz_results_N40_chi1000_iter239.csv",
        "xxz_bond/xxz_results_N50_chi1000_iter269.csv",
        "xxz_bond/xxz_results_N60_chi1000_iter179.csv",
        "xxz_bond/xxz_results_N80_chi1000_iter139.csv",
        "xxz_bond/xxz_results_N100_chi1000_iter139.csv"
    ]
    dfs_large = []
    for f in datafiles:
        dfs_large.append(pd.read_csv(f))
    df_large = pd.concat(dfs_large)
    print(df_large.head())
    return (df_large,)


@app.cell
def _(df_large, plt, sns):
    fig, ax = plt.subplots(1, 4, figsize=(10., 4.))
    sns.lineplot(data=df_large, x="iteration", y="abs_error", hue="N", ax=ax[0])
    sns.lineplot(data=df_large, x="iteration", y="time", hue="N", ax=ax[1])
    sns.lineplot(data=df_large, x="iteration", y="bond_dim", hue="N", ax=ax[2])
    sns.lineplot(data=df_large, x="iteration", y="depths", hue="N", ax=ax[3])
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Absolute energy error")
    ax[1].set_ylabel("Iteration time (ns)")
    ax[2].set_ylabel("Bond dimension")
    ax[3].set_ylabel("Circuit depth")
    fig.tight_layout()
    # plt.show()
    plt.savefig("scaling_study_results.pdf")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
