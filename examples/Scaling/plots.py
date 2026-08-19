import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    return pd, plt


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


if __name__ == "__main__":
    app.run()
