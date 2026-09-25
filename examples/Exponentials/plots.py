import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Time the default tensor network implementation
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alternate method to compute states
    """)
    return


@app.cell
def _(pd):
    df_alt = pd.read_csv("alternate_expm_results.csv")
    print(df_alt.head())
    return (df_alt,)


@app.cell
def _(df_alt, plt):
    fig2, ax2 = plt.subplots(1, 2)
    ax2[0].errorbar(df_alt["depths"], df_alt["avg_times_tn"], yerr=df_alt["std_times_tn"], label="Default")
    ax2[0].errorbar(df_alt["depths"], df_alt["avg_times_alt"], yerr=df_alt["std_times_alt"], label="Alternative")
    ax2[0].legend()
    ax2[0].set_xlabel("Depth")
    ax2[0].set_ylabel("Time (ns)")

    ax2[1].errorbar(df_alt["depths"], df_alt["avg_infidelity"], yerr=df_alt["std_infidelity"])
    ax2[1].set_xlabel("Depth")
    ax2[1].set_ylabel("Infidelity")
    # ax2[1].set_yscale("log")

    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exponentials from the trigonometric identity

    For GSD generators $A^3 = -A$, so $e^{cA} = I + \sin(c) A + (1 - \cos(c)) A^2$.
    Each exponential is applied with two MPO-MPS products and one compression, instead of simulating a circuit.
    """)
    return


@app.cell
def _(pd):
    df_trig = pd.read_csv("trig_mpo_expm_results.csv")
    print(df_trig.head())
    return (df_trig,)


@app.cell
def _(df_trig, plt):
    fig3, ax3 = plt.subplots(1, 2)
    ax3[0].errorbar(df_trig["depths"], df_trig["avg_times_la"], yerr=df_trig["std_times_la"], label="LinAlg")
    ax3[0].errorbar(df_trig["depths"], df_trig["avg_times_tn"], yerr=df_trig["std_times_tn"], label="TensorNet (circuit)")
    ax3[0].errorbar(df_trig["depths"], df_trig["avg_times_trig"], yerr=df_trig["std_times_trig"], label="TensorNet (trig. MPO)")
    ax3[0].legend()
    ax3[0].set_xlabel("Depth")
    ax3[0].set_ylabel("Time (ns)")
    ax3[0].set_yscale("log")

    ax3[1].errorbar(df_trig["depths"], df_trig["avg_infidelity"], yerr=df_trig["std_infidelity"])
    ax3[1].set_xlabel("Depth")
    ax3[1].set_ylabel("Infidelity (circuit vs. trig. MPO)")

    fig3.tight_layout()
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Trigonometric identity for all pools

    `tn_expm_mult_state` of GSD, SD, QE and CEO (and their subclasses) now uses the identity above.
    Both tensor network methods are compared to the exact (sparse) state.
    """)
    return


@app.cell
def _(pd):
    df_pools = pd.read_csv("trig_tn_expm_pools_results.csv")
    print(df_pools.groupby("pool")["speedup"].describe())
    return (df_pools,)


@app.cell
def _(df_pools, np, plt):
    fig4, ax4 = plt.subplots(1, 2, figsize=(10, 4))
    for pool_name in np.unique(df_pools["pool"]):
        df_p = df_pools[df_pools["pool"] == pool_name]
        ax4[0].plot(df_p["depths"], df_p["speedup"], label=pool_name)
        line, = ax4[1].plot(df_p["depths"], df_p["avg_infidelity_trig"], label=f"{pool_name} (trig.)")
        ax4[1].plot(df_p["depths"], df_p["avg_infidelity_circuit"], "--", color=line.get_color())
    ax4[0].set_xlabel("Depth")
    ax4[0].set_ylabel("Speedup (circuit time / trig. time)")
    ax4[0].legend()
    ax4[1].set_xlabel("Depth")
    ax4[1].set_ylabel("Infidelity vs. exact (dashed: circuit)")
    ax4[1].set_yscale("log")
    fig4.tight_layout()
    fig4
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
