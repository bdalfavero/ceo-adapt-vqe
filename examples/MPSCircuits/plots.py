import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt

    return pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## XXZ model
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("mps_to_circuit_results.csv")
    print(df.head())
    return (df,)


@app.cell
def _(pd):
    df_adapt = pd.read_csv("xxz_adapt_bond_results.csv")
    print(df_adapt.head())
    return (df_adapt,)


@app.cell
def _(df, df_adapt, plt):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    right_ax = ax[0].twinx()
    ax[0].plot(df["chi"], df["error"], color="blue", linestyle="-", label="DMRG")
    ax[0].plot(df["chi"], df["exact_error"], color="blue", linestyle="--", label="Exact MPS")
    ax[0].plot(df["chi"], df["approx_error"], color="blue", linestyle="-.", label="Approximate MPS")
    right_ax.plot(df["chi"], df["depths"], color="red", label="Exact")
    right_ax.plot(df["chi"], df["approx_depths"], linestyle="--", color="red", label="Approximate")
    ax[0].legend(loc="upper right")
    right_ax.legend(loc="center right")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Energy Error", color="blue")
    right_ax.set_ylabel("Circuit depth", color="red")
    ax[0].set_xlabel("Bond Dimension")
    ax[0].set_title("mps-to-circuit")

    right_ax2 = ax[1].twinx()
    ax[1].plot(df_adapt.index, df_adapt["error"], color="blue")
    right_ax2.plot(df_adapt.index, df_adapt["depths"], color="red")
    # right_ax.legend()
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Energy Error", color="blue")
    right_ax2.set_ylabel("Circuit depth", color="red")
    ax[1].set_xlabel("Iteration")
    ax[1].set_title(r"ADAPT ($\chi$=32)")

    fig.tight_layout()
    plt.show()
    return


@app.cell
def _(df, df_adapt, plt):
    # Scatterplot of energy vs. Circuit Depth for both
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["depths"], df["error"], label="mps-to-circuit")
    ax2.scatter(df_adapt["depths"], df_adapt["error"], label="ADAPT")
    ax2.legend()
    ax2.set_yscale("log")
    # ax2.set_xscale("log")
    ax2.set_ylabel("Energy Error")
    ax2.set_xlabel("Circuit Depth")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hydrogen chains
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(pd):
    df_hchain = pd.read_csv("hchain_mps_to_circuit_results.csv")
    print(df_hchain.head())
    return (df_hchain,)


@app.cell
def _(pd):
    df_hc_ad = pd.read_csv("hchain_adapt_bond_results.csv")
    print(df_hc_ad.head())
    return (df_hc_ad,)


@app.cell
def _(df_hc_ad, df_hchain, plt):
    fig_hchain, ax_hchain = plt.subplots(1, 2, figsize=(10, 5))

    right_ax_hchain = ax_hchain[0].twinx()
    ax_hchain[0].plot(df_hchain["chi"], df_hchain["error"], color="blue")
    right_ax_hchain.plot(df_hchain["chi"], df_hchain["depths"], color="red", label="Exact")
    right_ax_hchain.plot(df_hchain["chi"], df_hchain["approx_depths"], linestyle="--", color="red", label="Approximate")
    right_ax_hchain.legend(loc="center right")
    ax_hchain[0].set_yscale("log")
    ax_hchain[0].set_ylabel("Energy Error", color="blue")
    right_ax_hchain.set_ylabel("Circuit depth", color="red")
    ax_hchain[0].set_xlabel("Bond Dimension")
    ax_hchain[0].set_title("mps-to-circuit")

    right_ax2_hchain = ax_hchain[1].twinx()
    ax_hchain[1].plot(df_hc_ad.index, df_hc_ad["error"], color="blue")
    right_ax2_hchain.plot(df_hc_ad.index, df_hc_ad["depths"], color="red")
    # right_ax2_hchain.legend()
    ax_hchain[1].set_yscale("log")
    ax_hchain[1].set_ylabel("Energy Error", color="blue")
    right_ax2_hchain.set_ylabel("Circuit depth", color="red")
    ax_hchain[1].set_xlabel("Iteration")
    ax_hchain[1].set_title(r"ADAPT ($\chi$=100)")

    fig_hchain.tight_layout()
    plt.show()
    return


@app.cell
def _(df_hc_ad, df_hchain, plt):
    # Scatterplot of energy vs. Circuit Depth for both
    fig2_hchain, ax2_hchain = plt.subplots()
    ax2_hchain.scatter(df_hchain["depths"], df_hchain["error"], label="mps-to-circuit")
    ax2_hchain.scatter(df_hc_ad["depths"], df_hc_ad["error"], label="ADAPT")
    ax2_hchain.legend()
    ax2_hchain.set_yscale("log")
    # ax2_hchain.set_xscale("log")
    ax2_hchain.set_ylabel("Energy Error")
    ax2_hchain.set_xlabel("Circuit Depth")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
