import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def parse_data(data):
    if data["Type"] == "BasicTagged":
        data["Counters"] = np.array(data["Counters"])
        data["Sums"] = np.array(data["Sums"])
        data["SumsSquared"] = np.array(data["SumsSquared"])
    elif data["Type"] in ["Hist1dLog", "Hist1dLin"]:
        data["Count"] = np.array(data["Count"])
        data["Lower"] = np.array(data["Lower"])
        data["Upper"] = np.array(data["Upper"])
    elif data["Type"] in ["TaggedHist1dLog", "TaggedHist1dLin"]:
        data["Counts"] = np.array(data["Counts"])
        data["Lower"] = np.array(data["Lower"])
        data["Upper"] = np.array(data["Upper"])
    else:
        raise ValueError(f"Unknown stats type {data['Type']}")


def load_stats(path):
    all_data = json.loads(Path(path).read_text())
    for key in all_data:
        parse_data(all_data[key])
    return all_data


def extract_center(data):
    lower = data["Lower"]
    upper = data["Upper"]
    center = np.zeros(len(lower))
    center[1:-1] = (lower[1:-1] + upper[1:-1]) / 2
    center[0] = upper[0]
    center[-1] = lower[-1]

    return center


def plot_with_std(x, mean, std, *, axis: plt.Axes = None, **kwargs):
    if axis is None:
        axis = plt.gca()
    lines = axis.plot(x, mean, **kwargs)
    color = lines[0].get_color()
    axis.plot(x, mean + std, "--", color=color)
    axis.plot(x, mean - std, "--", color=color)
    axis.fill_between(x, mean+std, mean-std, color=color, alpha=0.2)


def plot_tagged_hist_over_tag(data, axis, **kwargs):
    counts = data["Counts"]
    centre = extract_center(data)

    # This is only approximate, but for a quick visualization I guess it's OK
    norm = np.sum(counts, axis=1)
    mean = np.sum(counts * centre[None, :], axis=1) / norm
    ssq = np.sum(counts * centre[None, :]**2, axis=1) / norm
    std = np.sqrt(ssq - mean**2)
    plot_with_std(np.arange(len(mean)), mean, std, axis=axis, **kwargs)

    if "Unit" in data:
        axis.set_ylabel(data["Unit"])


def quick_viz(data, fig: plt.Figure = None, **kwargs):
    if fig is None:
        fig = plt.figure()      # type: plt.Figure

    if data["Type"] == "BasicTagged":
        count = data["Counters"]
        mean = data["Sums"] / count
        std = np.sqrt(data["SumsSquared"] / count - (data["Sums"] / count)**2)

        axis = fig.axes[0]
        plot_with_std(np.arange(len(mean)), mean, std, axis=axis, **kwargs)
        if "Unit" in data:
            axis.set_ylabel(data["Unit"])

    elif data["Type"] in ["Hist1dLog", "Hist1dLin"]:
        count = data["Count"]
        centre = extract_center(data)

        axis = fig.axes[0]

        axis.plot(centre, count, **kwargs)
        if data["Type"] == "Hist1dLog":
            axis.set_xscale("log")
        if "Unit" in data:
            axis.set_xlabel(data["Unit"])
        axis.ylim((0, None))

    elif data["Type"] in ["TaggedHist1dLin", "TaggedHist1dLog"]:
        counts = data["Counts"]
        centre = extract_center(data)

        if len(fig.axes) < 2:
            axes = fig.subplots(1, 2)
        else:
            axes = fig.axes

        # first attempt: reduce over tag
        axis = axes[0]
        axis.plot(centre, np.sum(counts, axis=0), **kwargs)
        if data["Type"] == "TaggedHist1dLog":
            axis.set_xscale("log")
        if "Unit" in data:
            axis.set_xlabel(data["Unit"])
        axis.set_ylim((0, None))

        plot_tagged_hist_over_tag(data, axes[1], **kwargs)
    else:
        assert False

    return fig

