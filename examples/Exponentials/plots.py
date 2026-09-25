import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("exponential_timing.csv")
    print(df.head())
    return (df,)


@app.cell
def _(df, np, plt):
    fig, ax = plt.subplots()

    df_la = df.query("method == 'LinAlg'")
    ax.errorbar(df_la["depths"], df_la["avg_times"], yerr=df_la["std_times"], label="LinAlg")

    df_tn = df.query("method == 'TensorNet'")
    for chi in np.unique(df_tn["chi"]):
        df_tn_chi = df_tn[df_tn["chi"] == chi]
        ax.errorbar(df_tn_chi["depths"], df_tn_chi["avg_times"], yerr=df_tn_chi["std_times"], label=f"TensorNet, chi={chi}")

    ax.legend()
    ax.set_xlabel("Ansatz depth")
    ax.set_ylabel("Time (ns)")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
