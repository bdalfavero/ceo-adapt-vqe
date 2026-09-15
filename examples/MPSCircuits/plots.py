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
    styles = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"), ((0, (3, 1, 1, 1)), "v")]
    for i, ((method, num_layers), df_group) in enumerate(
        df.groupby(["method", "num_layers"])
    ):
        df_group = df_group.sort_values("chi")
        label = f"{method} (num_layers={num_layers})" if method == "approximate" else method
        linestyle, marker = styles[i % len(styles)]
        ax[0].plot(
            df_group["chi"], df_group["error"],
            color="blue", linestyle=linestyle, marker=marker, markersize=4, label=label,
        )
        if method != "DMRG":
            right_ax.plot(
                df_group["chi"], df_group["posttrans_depth"],
                color="red", linestyle=linestyle, marker=marker, markersize=4,
            )
    ax[0].legend(loc="upper right")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Energy Error", color="blue")
    right_ax.set_ylabel("Circuit depth", color="red")
    ax[0].set_xlabel("Bond Dimension")
    ax[0].set_title("mps-to-circuit")

    right_ax2 = ax[1].twinx()
    ax[1].plot(df_adapt.index, df_adapt["error"], color="blue")
    right_ax2.plot(df_adapt.index, df_adapt["depths"], color="red", label="Post-transpilation")
    right_ax2.plot(df_adapt.index, df_adapt["pretrans_depths"], color="red", linestyle="--", label="Pre-transpilation")
    right_ax2.legend()
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
    scatter_markers = ["o", "s", "^", "D", "v"]
    for j, ((scatter_method, scatter_num_layers), scatter_df_group) in enumerate(
        df.groupby(["method", "num_layers"])
    ):
        scatter_label = (
            f"{scatter_method} (num_layers={scatter_num_layers})"
            if scatter_method == "approximate"
            else scatter_method
        )
        if scatter_method != "DMRG":
            ax2.scatter(
                scatter_df_group["posttrans_depth"], scatter_df_group["error"],
                color="tab:blue", marker=scatter_markers[j % len(scatter_markers)], label=scatter_label,
            )
    ax2.scatter(df_adapt["depths"], df_adapt["error"], color="tab:red", marker="x", label="ADAPT")
    ax2.scatter(df_adapt["pretrans_depths"], df_adapt["error"], color="tab:red", marker="2", label="ADAPT (pre-transpilation)")
    ax2.legend()
    ax2.set_yscale("log")
    # ax2.set_xscale("log")
    ax2.set_ylabel("Energy Error")
    ax2.set_xlabel("Circuit Depth")
    plt.show()
    return


@app.cell
def _(df, df_adapt, plt):
    def _():
        # Make a scatterplot of pre-transpilation vs. post-transpilation depth for each method.
        fig3, ax3 = plt.subplots()
        scatter_markers = ["o", "s", "^", "D", "v"]
        for j2, ((scatter_method, scatter_num_layers), scatter_df_group) in enumerate(
            df.groupby(["method", "num_layers"])
        ):
            scatter_label = (
                f"{scatter_method} (num_layers={scatter_num_layers})"
                if scatter_method == "approximate"
                else scatter_method
            )
            if scatter_method != "DMRG":
                ax3.scatter(
                    scatter_df_group["pretrans_depth"], scatter_df_group["posttrans_depth"],
                    color="tab:blue", marker=scatter_markers[j2 % len(scatter_markers)], label=scatter_label,
                )
        ax3.scatter(df_adapt["pretrans_depths"], df_adapt["depths"], color="tab:red", marker="x", label="ADAPT")
        ax3.legend()
        # ax3.set_yscale("log")
        # ax3.set_xscale("log")
        ax3.set_ylabel("Post-transpile Depth")
        ax3.set_xlabel("Pre-transpile Depth")
        return plt.show()


    _()
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
