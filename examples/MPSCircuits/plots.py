import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt

    return pd, plt


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
    ax[0].plot(df["chi"], df["error"], color="blue")
    right_ax.plot(df["chi"], df["depths"], color="red", label="Exact")
    right_ax.plot(df["chi"], df["approx_depths"], linestyle="--", color="red", label="Approximate")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
