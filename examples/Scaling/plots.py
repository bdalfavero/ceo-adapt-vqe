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

    return np, os, pd, plt


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
