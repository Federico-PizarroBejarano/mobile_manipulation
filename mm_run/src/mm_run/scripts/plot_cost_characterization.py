"""Plot cost-characterization base-path figures for the paper.

Exports two TikZ/PDF/PNG figures with a shared world frame:
  - commanded twist channels (teleop_table_wipe/char)
  - base effort (teleop_table_wipe/char2)

Plot coordinates: u = -y_world (along-table, left→right), v = x_world
(across-table, robot below tables). Table footprints are drawn from the
table_wipe scene bounds.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

RESULTS = Path(__file__).resolve().parents[3] / "results" / "mpsf_experiments"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[4] / "docs" / "paper" / "figs"

# Shared axis box for both figures (plot frame after world→plot transform).
# Chosen to cover both wipe studies and both tables with equal aspect.
U_LIM = (-0.05, 2.70)  # -y_world
V_LIM = (-0.55, 1.60)  #  x_world
AXIS_WIDTH = "8.0cm"
AXIS_HEIGHT = "6.15cm"  # ≈ 8.0 * (V span / U span)
DOWNSAMPLE = 4

# Table footprints in world (x, y); from scenes/table_wipe.yaml comments.
TABLES_WORLD = (
    # (xmin, xmax, ymin, ymax, label)
    (0.6993, 1.4867, -1.5240, 0.0000, "Table 1"),
    (0.7946, 1.3915, -2.5273, -1.5240, "Table 2"),
)

TWIST_PROFILES = {
    (10.0, 10.0, 10.0, 10.0, 10.0, 10.0): "full",
    (10.0, 10.0, 10.0, 0.1, 0.1, 0.1): "moderate",
    (10.0, 10.0, 10.0, 0.01, 0.01, 0.01): "weak",
}
TWIST_ORDER = ["full", "moderate", "weak"]
TWIST_LABELS = {
    "full": "Full twist",
    "moderate": "Moderate angular",
    "weak": "Weak angular",
}
TWIST_COLORS = {
    "full": "#1b4f72",
    "moderate": "#48c78e",  # lighter green; distinguishable from navy
    "weak": "#b9770e",
}

BASE_PROFILES = {
    (0.1, 0.1, 0.1): "all_low",
    (0.1, 0.1, 10.0): "yaw",
    (10.0, 0.1, 10.0): "yaw_x",
    (10.0, 10.0, 10.0): "all_high",
}
BASE_ORDER = ["all_low", "yaw", "yaw_x", "all_high"]
BASE_LABELS = {
    "all_low": "All low",
    "yaw": "Raise yaw",
    "yaw_x": r"Raise yaw+$x$",
    "all_high": "All high",
}
BASE_COLORS = {
    "all_low": "#1b4f72",
    "yaw": "#48c78e",  # lighter green; distinguishable from navy
    "yaw_x": "#b9770e",
    "all_high": "#922b21",
}


def _as_float_tuple(values) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def world_to_plot(xy: np.ndarray) -> np.ndarray:
    """Map world (x, y) → plot (u, v) = (-y, x)."""
    xy = np.asarray(xy, dtype=float)
    return np.column_stack([-xy[:, 1], xy[:, 0]])


def base_path_length_m(base_pose: np.ndarray) -> float:
    xy = np.asarray(base_pose, dtype=float)[:, :2]
    if xy.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def downsample(xy: np.ndarray, stride: int = DOWNSAMPLE) -> np.ndarray:
    if xy.shape[0] <= 2:
        return xy
    idx = np.arange(0, xy.shape[0], stride)
    if idx[-1] != xy.shape[0] - 1:
        idx = np.append(idx, xy.shape[0] - 1)
    return xy[idx]


def load_trials(root: Path, profile_fn) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for trial_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg_path = trial_dir / "control" / "config.yaml"
        npz_path = trial_dir / "control" / "data.npz"
        if not cfg_path.exists() or not npz_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        label = profile_fn(cfg)
        if label is None:
            continue
        data = np.load(npz_path)
        base_pose = np.asarray(data["base_pose"], dtype=float)
        groups.setdefault(label, []).append(
            {
                "dir": trial_dir.name,
                "base_path_m": base_path_length_m(base_pose),
                "xy_plot": downsample(world_to_plot(base_pose[:, :2])),
            }
        )
    return groups


def twist_profile(cfg) -> str | None:
    qk = _as_float_tuple(cfg["controller"]["cost_params"]["EEVel"]["Qk"])
    return TWIST_PROFILES.get(qk)


def base_profile(cfg) -> str | None:
    qvb = _as_float_tuple(cfg["controller"]["cost_params"]["ControlEffort"]["Qvb"])
    return BASE_PROFILES.get(qvb)


def _representative(group: list[dict]) -> dict:
    """Pick the trial whose base-path length is closest to the group mean.

    This is a single real trajectory (not an average path), chosen so the bold
    overlay is typical of the condition's path length rather than an outlier.
    """
    paths = np.array([t["base_path_m"] for t in group])
    return group[int(np.argmin(np.abs(paths - paths.mean())))]


def draw_tables(ax):
    for xmin, xmax, ymin, ymax, label in TABLES_WORLD:
        # Corners in world → plot; rectangle axis-aligned in plot frame because
        # tables are axis-aligned in world and the map is a 90° rotation.
        corners = world_to_plot(
            np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        )
        u0, u1 = corners[:, 0].min(), corners[:, 0].max()
        v0, v1 = corners[:, 1].min(), corners[:, 1].max()
        ax.add_patch(
            Rectangle(
                (u0, v0),
                u1 - u0,
                v1 - v0,
                fill=True,
                facecolor="#d5d8dc",
                edgecolor="#566573",
                linewidth=1.0,
                zorder=1,
            )
        )
        ax.text(
            0.5 * (u0 + u1),
            0.5 * (v0 + v1),
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#2c3e50",
            zorder=2,
        )


def plot_paths(ax, trials: dict, order: list[str], colors: dict, labels: dict):
    draw_tables(ax)
    for name in order:
        color = colors[name]
        group = trials[name]
        for trial in group:
            ax.plot(
                trial["xy_plot"][:, 0],
                trial["xy_plot"][:, 1],
                color=color,
                alpha=0.25,
                linewidth=0.9,
                zorder=3,
                solid_capstyle="round",
                label="_nolegend_",
            )
        # Label only the bold representative so tikzplotlib emits legend entries.
        rep = _representative(group)
        ax.plot(
            rep["xy_plot"][:, 0],
            rep["xy_plot"][:, 1],
            color=color,
            linewidth=2.0,
            zorder=4,
            solid_capstyle="round",
            label=labels[name],
        )
        ax.plot(
            rep["xy_plot"][0, 0],
            rep["xy_plot"][0, 1],
            "o",
            color=color,
            markersize=3.5,
            zorder=5,
            label="_nolegend_",
        )
    ax.set_xlim(*U_LIM)
    ax.set_ylim(*V_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Along-table $-y$ [m]")
    ax.set_ylabel(r"Across-table $x$ [m]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", zorder=0)
    ax.set_axisbelow(True)
    # Sit over the empty table region so the legend does not cover paths.
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def _patch_tikzplotlib_webcolors():
    import webcolors

    if hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
        return
    webcolors.CSS3_HEX_TO_NAMES = {
        webcolors.name_to_hex(name): name for name in webcolors.names("css3")
    }


def _table_tikz() -> str:
    """Axis-aligned table rectangles/labels in plot coordinates."""
    chunks = []
    for xmin, xmax, ymin, ymax, label in TABLES_WORLD:
        corners = world_to_plot(
            np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        )
        u0, u1 = corners[:, 0].min(), corners[:, 0].max()
        v0, v1 = corners[:, 1].min(), corners[:, 1].max()
        uc = 0.5 * (u0 + u1)
        vc = 0.5 * (v0 + v1)
        chunks.append(
            "\\fill[gainsboro213216220, draw=dimgrey86101115, line width=0.6pt]\n"
            f"  (axis cs:{u0:.4f},{v0:.4f}) rectangle (axis cs:{u1:.4f},{v1:.4f});\n"
            f"\\node[font=\\scriptsize, text=darkslategrey446280]"
            f" at (axis cs:{uc:.4f},{vc:.4f}) {{{label}}};"
        )
    return "\n".join(chunks) + "\n"


def save_tikz(figure, destination: Path):
    import tikzplotlib

    _patch_tikzplotlib_webcolors()
    legends = list(figure.legends)
    legends.extend(legend for axis in figure.axes if (legend := axis.get_legend()))
    for legend in legends:
        if not hasattr(legend, "_ncol"):
            legend._ncol = legend._ncols
    destination.parent.mkdir(parents=True, exist_ok=True)
    tikzplotlib.save(
        destination,
        figure=figure,
        encoding="utf-8",
        textsize=9,
        include_disclaimer=False,
        axis_width=AXIS_WIDTH,
        axis_height=AXIS_HEIGHT,
        extra_axis_parameters=[
            r"label style={font=\small}",
            r"xlabel style={font=\small}",
            r"ylabel style={font=\small}",
            r"tick label style={font=\small}",
            "axis equal image",
        ],
    )
    text = destination.read_text()
    text = re.sub(
        r"grid style=\{([A-Za-z0-9]+)\}",
        r"grid style={color=\1}",
        text,
    )
    # Force shared limits explicitly (tikzplotlib sometimes expands them).
    text = re.sub(r"xmin=[^,\]]+", f"xmin={U_LIM[0]}", text, count=1)
    text = re.sub(r"xmax=[^,\]]+", f"xmax={U_LIM[1]}", text, count=1)
    text = re.sub(r"ymin=[^,\]]+", f"ymin={V_LIM[0]}", text, count=1)
    text = re.sub(r"ymax=[^,\]]+", f"ymax={V_LIM[1]}", text, count=1)
    text = text.replace(
        r"xlabel={Along-table \(\displaystyle -y\) [m]}",
        r"xlabel={Along-table $-y$ [m]}",
    )
    text = text.replace(
        r"ylabel={Across-table \(\displaystyle x\) [m]}",
        r"ylabel={Across-table $x$ [m]}",
    )
    # Replace whatever legend style tikzplotlib emitted with a compact one
    # over the empty table region (paths sit along the bottom).
    legend_style = (
        r"legend style={font=\scriptsize, draw=none, fill=none, "
        r"at={(0.02,0.59)}, anchor=north west, inner sep=1pt, row sep=-2pt},"
    )
    text = re.sub(
        r"legend style=\{(?:[^{}]|\{[^{}]*\})*\},",
        lambda _match: legend_style,
        text,
        count=1,
    )
    # Drop matplotlib-exported (tiny) table labels; inject filled rectangles.
    text = re.sub(
        r"\\draw \(axis cs:[^)]+\) node\[[^\]]*\]\{Table [12]\};\n?",
        "",
        text,
    )
    text = text.replace("]\n\\addplot", "]\n" + _table_tikz() + "\\addplot", 1)
    destination.write_text(text)


def build_figure(trials, order, colors, labels, out_stem: Path):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.9))
    plot_paths(ax, trials, order, colors, labels)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white"
    )
    save_tikz(fig, out_stem.with_suffix(".tex"))
    plt.close(fig)
    print(f"Wrote {out_stem.with_suffix('.pdf')}")
    print(f"Wrote {out_stem.with_suffix('.png')}")
    print(f"Wrote {out_stem.with_suffix('.tex')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    args = parser.parse_args()

    twist = load_trials(args.results_root / "teleop_table_wipe" / "char", twist_profile)
    base = load_trials(args.results_root / "teleop_table_wipe" / "char2", base_profile)

    for name, groups, expected in (
        ("twist", twist, TWIST_ORDER),
        ("base", base, BASE_ORDER),
    ):
        for key in expected:
            n = len(groups.get(key, []))
            if n != 5:
                raise SystemExit(f"{name}/{key}: expected 5 trials, found {n}")

    build_figure(
        twist,
        TWIST_ORDER,
        TWIST_COLORS,
        TWIST_LABELS,
        args.out_dir / "cost_twist_paths",
    )
    build_figure(
        base,
        BASE_ORDER,
        BASE_COLORS,
        BASE_LABELS,
        args.out_dir / "cost_base_paths",
    )


if __name__ == "__main__":
    main()
