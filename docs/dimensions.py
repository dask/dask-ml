import numpy as np

import matplotlib.pyplot as plt


def draw_brace(ax, xspan, text):
    """Draws an annotated brace on the axes."""
    # https://stackoverflow.com/a/53383764/1889400
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan / xax_span * 100) * 2 + 1  # guaranteed uneven
    beta = 300.0 / xax_span  # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[: resolution // 2 + 1]
    y_half_brace = 1 / (1.0 + np.exp(-beta * (x_half - x_half[0]))) + 1 / (
        1.0 + np.exp(-beta * (x_half - x_half[-1]))
    )
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (0.05 * y - 0.01) * yspan  # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color="black", lw=3)

    ax.text(
        (xmax + xmin) / 2,
        ymin + 0.07 * yspan,
        text,
        ha="center",
        va="bottom",
        fontsize=20,
    )


if __name__ == "__main__":
    plt.xkcd()
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.rcParams["font.size"] = 30
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.plot(
        (xlim[0] + 0.02, xlim[1] - 0.02),
        [0.49, 0.49],
        linestyle="dashed",
        lw=4,
        color="C0",
    )
    ax.plot(
        [0.5, 0.5],
        (ylim[0] + 0.02, ylim[1] - 0.02),
        linestyle="dashed",
        lw=4,
        color="C0",
    )

    ax.text(0.06, 0.8, "COMPUTE BOUND")
    ax.text(
        0.06, 0.65, "- Grid Search\n- Random Forest\n- cross_val_score\n…", fontsize=16
    )

    ax.text(0.6, 0.25, "MEMORY BOUND")

    ax.set(
        ylabel="MODEL SIZE",
        xticks=[],
        yticks=[],
        xlim=xlim,
        ylim=ylim,
        title="DIMENSIONS OF SCALE",
    )
    ax.set_xlabel(xlabel="DATA SIZE", labelpad=10, loc="right", fontsize=20)
    ax.set_ylabel("MODEL SIZE", labelpad=10, fontsize=20)

    draw_brace(ax, (0.01, 0.49), "FITS IN RAM")
    plt.savefig("source/images/dimensions_of_scale.svg")
    plt.savefig("source/images/dimensions_of_scale.png")
