import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PARTICIPANTS = ["ali", "mo", "wenda", "raghav", "sebas", "sat", "expert"]
LABELS = ["P1", "P2", "P3", "P4", "P5", "P6", "Expert"]
# Extra gap before the expert so that group stays visually separate.
POSITIONS = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.6])
TASKS = {"tray": "Tray carrying", "table_wipe": "Table wiping"}
APPROACHES = {"direct": "Direct teleop    ", "mpsf": "MPSF    "}
COLORS = {"direct": "#8B3B32", "mpsf": "#348432"}
AGGREGATE_POSITIONS = np.arange(len(TASKS), dtype=float)
AGGREGATE_OFFSET = 0.18
AGGREGATE_WIDTH = 0.28
EVENTS = ["collisions", "protective_stops", "singularities", "bottle_drops"]
OUTPUT = None


def default_metrics_csv() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "results"
        / "mpsf_experiments"
        / "user_study_metrics.csv"
    )


def _has_events(trials) -> bool:
    return all(event in trials.columns for event in EVENTS)


def load_trials(source: Path):
    trials = pd.read_csv(source)
    missing = {
        "task",
        "user",
        "backend",
        "trial",
        "duration_s",
        "mode_switch_count",
    } - set(trials.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    trials = trials.copy()
    keys = ["task", "user", "backend", "trial"]
    if trials.duplicated(keys).any():
        dupes = trials.loc[trials.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicate trial identifiers:\n{dupes}")
    counts = trials.groupby(keys[:3]).size()
    if counts.empty or not counts.eq(3).all():
        bad = counts[counts.ne(3)]
        raise ValueError(
            "Expected 3 trials per participant, task, and approach:\n"
            f"{bad if len(bad) else 'no matching rows'}"
        )
    if trials.duration_s.isna().any() or trials.duration_s.le(0).any():
        raise ValueError("Completion times must be positive and present")
    if trials.mode_switch_count.isna().any() or trials.mode_switch_count.lt(0).any():
        raise ValueError("Mode switch counts must be present and non-negative")
    missing_events = [event for event in EVENTS if event not in trials.columns]
    if missing_events:
        raise ValueError(
            f"{source} is missing incident columns: {missing_events}. "
            "Export the metrics CSV with user_study_incidents.csv beside it."
        )
    for event in EVENTS:
        applicable = (
            trials.task.eq("tray")
            if event == "bottle_drops"
            else pd.Series(True, index=trials.index)
        )
        if trials.loc[applicable, event].isna().any():
            raise ValueError(f"Missing applicable event counts: {event}")
        if trials.loc[applicable, event].lt(0).any():
            raise ValueError(f"Negative event counts: {event}")
    trials["incident_free"] = trials[EVENTS].fillna(0).eq(0).all(axis=1)
    trials["is_expert"] = trials.user.eq("expert")
    return trials


def style_axis(axis, title, ylabel):
    axis.set_title(title, loc="left", fontweight="bold", pad=12)
    axis.set_ylabel(ylabel)
    axis.set_xticks(POSITIONS, LABELS)
    axis.axvline(
        (POSITIONS[-2] + POSITIONS[-1]) / 2,
        color="0.75",
        linewidth=1,
        linestyle="--",
    )
    axis.set_xlim(POSITIONS[0] - 0.55, POSITIONS[-1] + 0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", zorder=0)
    axis.set_axisbelow(True)


def save_figure(figure, name):
    figure.savefig(OUTPUT / f"{name}.png", dpi=190, facecolor="white")
    figure.savefig(OUTPUT / f"{name}.pdf", facecolor="white")
    plt.close(figure)


def _patch_tikzplotlib_webcolors():
    """tikzplotlib still expects the webcolors 1.x color-name table."""
    import webcolors

    if hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
        return
    webcolors.CSS3_HEX_TO_NAMES = {
        webcolors.name_to_hex(name): name for name in webcolors.names("css3")
    }


def _legend_image(options: str, label: str) -> str:
    mark = "diamond*" if "diamond" in options else "square*"
    colors = re.findall(r"(?:sienna|forestgreen|black)\d+", options)
    color = colors[0] if colors else "black"
    return (
        "\\addlegendimage{only marks, mark=%s, mark size=2.6pt, "
        "mark options={fill=%s, draw=%s}}\n\\addlegendentry{%s}"
        % (mark, color, color, label)
    )


def save_tikz(figure, name):
    import tikzplotlib

    _patch_tikzplotlib_webcolors()
    legends = list(figure.legends)
    legends.extend(legend for axis in figure.axes if (legend := axis.get_legend()))
    for legend in legends:
        if not hasattr(legend, "_ncol"):
            legend._ncol = legend._ncols
    destination = (
        Path(__file__).resolve().parents[3]
        / "results"
        / "mpsf_experiments"
        / "user_study_plots"
        / f"{name}.tex"
    )
    tikzplotlib.save(
        destination,
        figure=figure,
        encoding="utf-8",
        textsize=10,
        include_disclaimer=False,
        # Width is the axis box. The gap clears a single-line ylabel.
        axis_width="5.95cm",
        axis_height="5.8cm",
        extra_axis_parameters=[
            r"label style={font=\small}",
            r"xlabel style={font=\small}",
        ],
        extra_groupstyle_parameters=["horizontal sep=1.55cm"],
    )
    text = destination.read_text()
    text = re.sub(
        r"grid style=\{([A-Za-z0-9]+)\}",
        r"grid style={color=\1}",
        text,
    )
    # Empty legend series become "nan nan" tables, which pgfplots rejects.
    text = re.sub(
        r"\\addplot \[([^\]]+)\]\ntable \{\%\nnan nan\n\};\n\\addlegendentry\{([^}]+)\}",
        lambda match: _legend_image(match.group(1), match.group(2)),
        text,
    )
    text = text.replace(
        "xticklabels={Tray carrying,Table wiping}",
        r"xticklabels={Tray\\carrying,Table\\wiping}, xticklabel style={align=center, font=\small}",
    )
    text = re.sub(
        r"scale=0\.\d+,\n(  anchor=south,\n  text=\w+,\n  rotate=0\.0\n\]\{\d+/\d+\})",
        r"scale=0.95,\n\1",
        text,
    )
    text = text.replace(
        "draw=none,\n  fill=none\n}",
        "draw=none,\n  fill=none,\n  column sep=0.25em\n}",
    )
    text = text.replace(
        "ymax=1.25,",
        "enlarge y limits=false,\nymax=1.25,\nytick={0,0.25,0.5,0.75,1},",
    )
    # Frame transparency is stored as fill opacity, which also hides legend marks.
    text = text.replace("  fill opacity=0,\n", "")
    destination.write_text(
        "% Requires \\usetikzlibrary{plotmarks} in the preamble.\n" + text
    )


def plot_times(
    trials, incident_free_only=False, participants=None, approaches=None, name=None
):
    figure, axes = plt.subplots(1, 2, figsize=(2.3 * len(PARTICIPANTS) + 3.5, 5.0))
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.21, top=0.82, wspace=0.22)
    selected = trials[trials.incident_free] if incident_free_only else trials
    for axis, (task, title) in zip(axes, TASKS.items()):
        style_axis(axis, title, "Completion time [s]")
        axis.set_ylim(0, trials.loc[trials.task.eq(task), "duration_s"].max() * 1.18)
        for participant_index, participant in enumerate(PARTICIPANTS):
            if participants is not None and participant not in participants:
                continue
            for approach_index, approach in enumerate(APPROACHES):
                if approaches is not None and approach not in approaches:
                    continue
                subset = selected[
                    (selected.task == task)
                    & (selected.user == participant)
                    & (selected.backend == approach)
                ]
                center = POSITIONS[participant_index] + (
                    -0.18 if approach_index == 0 else 0.18
                )
                if subset.empty:
                    axis.text(center, 6, "n=0", ha="center", fontsize=9)
                    continue
                jitter = (subset.trial.to_numpy() - 2) * 0.06
                for location, (_, trial) in zip(center + jitter, subset.iterrows()):
                    axis.scatter(
                        location,
                        trial.duration_s,
                        marker="o" if trial.incident_free else "X",
                        s=48,
                        color=COLORS[approach],
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=3,
                    )
                mean = subset.duration_s.mean()
                axis.plot(
                    [center - 0.12, center + 0.12],
                    [mean, mean],
                    color=COLORS[approach],
                    linewidth=3,
                    zorder=4,
                )
                if incident_free_only:
                    axis.text(
                        center,
                        subset.duration_s.max() + axis.get_ylim()[1] * 0.035,
                        f"n={len(subset)}",
                        ha="center",
                        fontsize=9,
                        color=COLORS[approach],
                    )
    handles = [
        Line2D([0], [0], color=color, linewidth=3, label=APPROACHES[approach])
        for approach, color in COLORS.items()
    ]
    if _has_events(trials):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="0.35",
                label="Incident-free trial",
            )
        )
        if not incident_free_only:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    linestyle="none",
                    color="0.35",
                    label="Trial with incident(s)",
                )
            )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=len(handles),
        frameon=False,
    )
    if incident_free_only:
        figure.suptitle(
            "Incident-free trials only", y=0.97, fontsize=16, fontweight="bold"
        )
        figure.text(
            0.5,
            0.025,
            "Points = trials; horizontal bars = means. Expert shown separately.",
            ha="center",
            fontsize=10,
            color="0.35",
        )
    else:
        figure.subplots_adjust(top=0.93)
    save_figure(
        figure,
        name
        or (
            "completion_time_incident_free"
            if incident_free_only
            else "completion_time_all"
        ),
    )


def plot_time_reveals(trials):
    novices = [user for user in PARTICIPANTS if user != "expert"]
    plot_times(trials, participants=[], name="completion_time_axes")
    plot_times(
        trials,
        participants=novices,
        approaches=["direct"],
        name="completion_time_direct_participants",
    )
    plot_times(trials, participants=novices, name="completion_time_all_participants")


def plot_reliability(trials):
    figure, axes = plt.subplots(1, 2, figsize=(2.3 * len(PARTICIPANTS) + 3.5, 4.8))
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.22, top=0.81, wspace=0.22)
    for axis, (task, title) in zip(axes, TASKS.items()):
        style_axis(axis, title, "Incident-free trials")
        axis.set_ylim(0, 3.55)
        axis.set_yticks([0, 1, 2, 3])
        for approach_index, (approach, label) in enumerate(APPROACHES.items()):
            counts = [
                int(
                    trials.loc[
                        (trials.task == task)
                        & (trials.user == participant)
                        & (trials.backend == approach),
                        "incident_free",
                    ].sum()
                )
                for participant in PARTICIPANTS
            ]
            centers = POSITIONS + (-0.18 if approach_index == 0 else 0.18)
            axis.bar(
                centers,
                counts,
                width=0.32,
                color=COLORS[approach],
                label=label,
                zorder=3,
            )
            for center, count in zip(centers, counts):
                axis.text(center, count + 0.08, f"{count}/3", ha="center", fontsize=10)
    figure.suptitle(
        "Completion without recorded incidents", y=0.97, fontsize=16, fontweight="bold"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=2,
        frameon=False,
    )
    figure.text(
        0.5,
        0.03,
        "Incidents: collisions, protective stops, singularities, and bottle drops.",
        ha="center",
        fontsize=10,
        color="0.35",
    )
    save_figure(figure, "incident_free_completion")


def plot_direct_failure_types(trials):
    selected = trials[trials.backend.eq("direct") & ~trials.is_expert]
    event_labels = ["Collisions", "Protective\nstops", "Singularities", "Bottle drops"]
    task_colors = {"tray": "#8B3B32", "table_wipe": "#547A99"}
    figure, axis = plt.subplots(figsize=(10.0, 4.5))
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.94)
    event_positions = np.arange(len(EVENTS))
    summary_rows = []
    for task_index, (task, label) in enumerate(TASKS.items()):
        subset = selected[selected.task.eq(task)]
        centers = event_positions + (-0.19 if task_index == 0 else 0.19)
        counts = subset[EVENTS].sum().to_numpy()
        bars = axis.bar(
            centers,
            counts,
            width=0.34,
            color=task_colors[task],
            label=f"{label} ({len(subset)} trials)",
            zorder=3,
        )
        for event_index, (event, bar) in enumerate(zip(EVENTS, bars)):
            applicable = not (task == "table_wipe" and event == "bottle_drops")
            event_count = int(counts[event_index]) if applicable else None
            affected_trials = int(subset[event].gt(0).sum()) if applicable else None
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                str(event_count) if applicable else "N/A",
                ha="center",
                va="bottom",
                fontsize=11,
            )
            summary_rows.append(
                {
                    "task": task,
                    "failure_type": event,
                    "trial_count": len(subset),
                    "event_count": event_count,
                    "affected_trials": affected_trials,
                }
            )
    axis.set_ylabel("Recorded events")
    axis.set_xticks(event_positions, event_labels)
    axis.set_ylim(0, selected.groupby("task")[EVENTS].sum().max().max() + 1.5)
    axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9")
    axis.set_axisbelow(True)
    figure.legend(
        loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=2, frameon=False
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "direct_teleop_failure_types.csv", index=False
    )
    save_figure(figure, "direct_teleop_failure_types")


def _aggregate_center(task_index, approach_index):
    return AGGREGATE_POSITIONS[task_index] + (
        -AGGREGATE_OFFSET if approach_index == 0 else AGGREGATE_OFFSET
    )


def _style_aggregate_axis(axis, caption, ylabel):
    # The caption is the xlabel so it sits under the tick labels as (a), (b), (c).
    axis.set_xlabel(caption)
    axis.set_ylabel(ylabel)
    axis.set_xticks(AGGREGATE_POSITIONS, list(TASKS.values()))
    axis.set_xlim(AGGREGATE_POSITIONS[0] - 0.55, AGGREGATE_POSITIONS[-1] + 0.55)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", zorder=0)
    axis.set_axisbelow(True)


def _condition_values(trials, task, approach, column):
    values = trials.loc[
        trials.task.eq(task) & trials.backend.eq(approach), column
    ].to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No participant trials for {task} / {approach}")
    return values


def _draw_distribution(axis, trials, column, rng):
    for task_index, task in enumerate(TASKS):
        for approach_index, approach in enumerate(APPROACHES):
            values = _condition_values(trials, task, approach, column)
            center = _aggregate_center(task_index, approach_index)
            drawn = axis.boxplot(
                values,
                positions=[center],
                widths=AGGREGATE_WIDTH,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={"color": "0.1", "linewidth": 1.5},
                whiskerprops={"color": "0.25", "linewidth": 1},
                capprops={"color": "0.25", "linewidth": 1},
                boxprops={
                    "facecolor": COLORS[approach],
                    "edgecolor": COLORS[approach],
                    "alpha": 0.28,
                    "linewidth": 1,
                },
            )
            for artist in (
                drawn["boxes"] + drawn["whiskers"] + drawn["caps"] + drawn["medians"]
            ):
                artist.set_zorder(2)
            jitter = rng.uniform(-0.06, 0.06, size=values.size)
            axis.scatter(
                np.full(values.size, center) + jitter,
                values,
                s=16,
                color=COLORS[approach],
                alpha=0.9,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
            axis.plot(
                [center],
                [float(values.mean())],
                linestyle="none",
                marker="D",
                color="0.1",
                markersize=4.5,
                zorder=4,
            )
    axis.set_ylim(0, trials[column].max() * 1.12)


def _draw_incident_rates(axis, trials):
    for task_index, task in enumerate(TASKS):
        for approach_index, approach in enumerate(APPROACHES):
            subset = trials.loc[trials.task.eq(task) & trials.backend.eq(approach)]
            if subset.empty:
                raise ValueError(f"No participant trials for {task} / {approach}")
            center = _aggregate_center(task_index, approach_index)
            n = len(subset)
            k = int(subset.incident_free.sum())
            height = k / n
            axis.bar(
                center,
                height,
                width=AGGREGATE_WIDTH,
                color=COLORS[approach],
                zorder=3,
            )
            axis.text(
                center,
                height + 0.05,
                f"{k}/{n}",
                ha="center",
                va="bottom",
                fontsize=18,
                color="0.15",
            )
    axis.set_ylim(0, 1.25)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def plot_aggregate(trials):
    """Completion time, incident-free rate, and mode switches for participants."""
    selected = trials.loc[~trials.is_expert].copy()
    if selected.empty:
        raise ValueError("No participant trials to aggregate")
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 4.15))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.86, wspace=0.62)
    rng = np.random.default_rng(0)
    _draw_distribution(axes[0], selected, "duration_s", rng)
    _style_aggregate_axis(axes[0], "(a) Completion time", "Time [s]")
    _draw_incident_rates(axes[1], selected)
    _style_aggregate_axis(axes[1], "(b) Incident-free trials", "Fraction of trials")
    _draw_distribution(axes[2], selected, "mode_switch_count", rng)
    _style_aggregate_axis(axes[2], "(c) Mode switches", "Switches per trial")
    axes[2].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    # These series carry the legend labels. NaNs are not drawn. The legend sits
    # above the left panel.
    for approach, color in COLORS.items():
        axes[0].plot(
            [np.nan],
            [np.nan],
            linestyle="none",
            marker="s",
            color=color,
            markersize=7,
            label=APPROACHES[approach],
        )
    axes[0].plot(
        [np.nan],
        [np.nan],
        linestyle="none",
        marker="D",
        color="0.1",
        markersize=5,
        label="Mean",
    )
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.06),
        ncol=3,
        frameon=False,
        framealpha=0,
        borderaxespad=0,
        fontsize=9,
        handletextpad=0.6,
        columnspacing=2.0,
    )
    save_tikz(figure, "user_study_aggregate")
    save_figure(figure, "user_study_aggregate")


def summarize(trials):
    group_keys = ["task", "user", "backend"]
    summary = trials.groupby(group_keys).agg(
        trials=("trial", "size"),
        incident_free_trials=("incident_free", "sum"),
        mean_time_s=("duration_s", "mean"),
        std_time_s=("duration_s", "std"),
        mean_jerk=("mean_jerk", "mean"),
        mean_mode_switches=("mode_switch_count", "mean"),
    )
    summary = summary.join(
        trials[trials.incident_free]
        .groupby(group_keys)
        .duration_s.mean()
        .rename("incident_free_mean_time_s")
    )
    if _has_events(trials):
        summary = summary.join(trials.groupby(group_keys)[EVENTS].sum())
    summary.to_csv(OUTPUT / "participant_summary.csv")
    aggregate = trials.groupby(["is_expert", "task", "backend"]).agg(
        trials=("trial", "size"),
        incident_free_trials=("incident_free", "sum"),
        mean_time_s=("duration_s", "mean"),
        mean_jerk=("mean_jerk", "mean"),
        mean_mode_switches=("mode_switch_count", "mean"),
    )
    if _has_events(trials):
        aggregate = aggregate.join(
            trials.groupby(["is_expert", "task", "backend"])[EVENTS].sum()
        )
    aggregate.to_csv(OUTPUT / "aggregate_summary.csv")
    print(aggregate.round(3).to_string())


def main(argv=None):
    global OUTPUT
    parser = argparse.ArgumentParser(
        description="Plot user-study completion times from user_study_metrics.csv."
    )
    parser.add_argument("--csv", type=Path, default=default_metrics_csv())
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figure directory (default: next to the metrics CSV)",
    )
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        raise SystemExit(f"Missing metrics CSV: {args.csv}")
    OUTPUT = args.output or args.csv.parent / "user_study_plots"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42}
    )
    trials = load_trials(args.csv)
    summarize(trials)
    plot_time_reveals(trials)
    plot_times(trials)
    plot_times(trials, incident_free_only=True)
    plot_reliability(trials)
    plot_direct_failure_types(trials)
    plot_aggregate(trials)


if __name__ == "__main__":
    main()
