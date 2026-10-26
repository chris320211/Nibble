"""Draw the README charts from the recorded Colab T4 numbers.

Numbers are copied from the README tables; rerun after updating them:

    python scripts/plot_results.py

Writes light and dark PNGs to docs/ (README picks one per GitHub theme).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "docs"

# Fixed categorical order: this repo, bitsandbytes, FP16.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"fused": "#2a78d6", "bnb": "#eb6834", "fp16": "#1baf7a"},
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"fused": "#3987e5", "bnb": "#d95926", "fp16": "#199e70"},
    },
}
NAMES = {"fused": "this repo (INT4)", "bnb": "bitsandbytes NF4", "fp16": "FP16 cuBLAS"}

# TinyLlama decode tok/s, CUDA graphs, one session (2+2 prompt row).
DECODE_TOK_S = {"fp16": 88.5, "bnb": 118.1, "fused": 183.6}

# M = 1 per-layer GPU time in us, one session.
LAYER_SHAPES = ["attention\n2048 x 2048", "mlp up\n2048 x 5632", "mlp down\n5632 x 2048"]
LAYER_US = {
    "fp16": [39.8, 106.5, 99.9],
    "bnb": [19.9, 61.3, 61.1],
    "fused": [12.5, 28.3, 32.1],
}

# mlp up (2048 x 5632) per kernel version, each from its own session.
VERSIONS = ["v1\noriginal", "v2\n16-byte loads", "v3\nfast unpack", "v4\n4 rows/warp"]
VERSION_US = [100.8, 38.7, 37.1, 27.9]
VERSION_FP16_US = 97.4


def style_axes(ax, theme: dict, value_axis: str) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    baseline = "left" if value_axis == "x" else "bottom"
    ax.spines[baseline].set_visible(True)
    ax.spines[baseline].set_color(theme["axis"])
    ax.grid(axis=value_axis, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["text_secondary"], length=0, labelsize=10)


def new_figure(theme: dict, width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    return fig, ax


def title(fig, theme: dict, text: str, subtitle: str) -> None:
    fig.text(0.02, 0.95, text, color=theme["text"], fontsize=13, weight="bold", va="top")
    fig.text(0.02, 0.86, subtitle, color=theme["text_secondary"], fontsize=10, va="top")


def plot_decode(theme: dict, path: Path) -> None:
    fig, ax = new_figure(theme, 8, 2.8)
    order = ["fp16", "bnb", "fused"]
    values = [DECODE_TOK_S[key] for key in order]
    bars = ax.barh(
        [NAMES[key] for key in order],
        values,
        height=0.6,
        color=[theme["series"][key] for key in order],
    )
    for bar, value in zip(bars, values):
        ax.text(
            value + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f} tok/s",
            color=theme["text"],
            va="center",
            fontsize=10,
        )
    style_axes(ax, theme, "x")
    ax.tick_params(axis="y", colors=theme["text"], labelsize=10)
    ax.set_xlim(0, 215)
    ax.set_xlabel("decode tokens per second (higher is better)", color=theme["muted"])
    title(fig, theme, "TinyLlama-1.1B decode on a T4", "CUDA graphs for all three, one Colab session")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.74, bottom=0.2)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def plot_layers(theme: dict, path: Path) -> None:
    fig, ax = new_figure(theme, 8, 3.9)
    order = ["fused", "bnb", "fp16"]
    width = 0.26
    gap = 0.015  # surface gap between adjacent bars
    for index, key in enumerate(order):
        positions = [shape + (index - 1) * width for shape in range(len(LAYER_SHAPES))]
        bars = ax.bar(
            positions,
            LAYER_US[key],
            width=width - gap,
            color=theme["series"][key],
            label=NAMES[key],
        )
        for bar, value in zip(bars, LAYER_US[key]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.1f}",
                color=theme["text"],
                ha="center",
                fontsize=9,
            )
    style_axes(ax, theme, "y")
    ax.set_xticks(range(len(LAYER_SHAPES)), LAYER_SHAPES, color=theme["text"])
    ax.set_ylim(0, 120)
    ax.set_ylabel("us per call (lower is better)", color=theme["muted"])
    legend = ax.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        borderaxespad=0,
        fontsize=9,
        ncol=3,
    )
    for text in legend.get_texts():
        text.set_color(theme["text_secondary"])
    title(fig, theme, "One linear layer, one token (M = 1)", "GPU time per call, weights read from DRAM")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.7, bottom=0.16)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def plot_versions(theme: dict, path: Path) -> None:
    fig, ax = new_figure(theme, 8, 3.2)
    bars = ax.bar(range(len(VERSIONS)), VERSION_US, width=0.55, color=theme["series"]["fused"])
    for bar, value in zip(bars, VERSION_US):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}",
            color=theme["text"],
            ha="center",
            fontsize=10,
        )
    ax.axhline(VERSION_FP16_US, color=theme["series"]["fp16"], linewidth=2, linestyle="--")
    ax.text(
        len(VERSIONS) - 0.6,
        VERSION_FP16_US + 3,
        f"FP16 cuBLAS {VERSION_FP16_US:.0f}",
        color=theme["text_secondary"],
        ha="right",
        fontsize=9,
    )
    style_axes(ax, theme, "y")
    ax.set_xticks(range(len(VERSIONS)), VERSIONS, color=theme["text"])
    ax.set_ylim(0, 120)
    ax.set_ylabel("us per call (lower is better)", color=theme["muted"])
    title(fig, theme, "Kernel versions, mlp up layer (2048 x 5632)", "Each version measured in its own session")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.77, bottom=0.19)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for mode, theme in THEMES.items():
        plot_decode(theme, OUT_DIR / f"decode_{mode}.png")
        plot_layers(theme, OUT_DIR / f"layers_{mode}.png")
        plot_versions(theme, OUT_DIR / f"versions_{mode}.png")
    print(f"Wrote charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
