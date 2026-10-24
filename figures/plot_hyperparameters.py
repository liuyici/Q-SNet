"""Render the supplied sweep with parameter boxes from the shared configuration."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configured_indices(sweep, config):
    selected = {}
    for parameter in sweep["parameters"]:
        key = parameter["key"]
        value = config["common"][key]
        matches = [i for i, candidate in enumerate(parameter["values"])
                   if math.isclose(candidate, value)]
        if len(matches) != 1:
            raise ValueError(f"Configured {key}={value} is not a unique sweep point")
        selected[key] = matches[0]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--data", type=Path, default=ROOT / "figures/hyperparameter_sweep.json")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/hyperparameter_sweep.png")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sweep = json.loads(args.data.read_text(encoding="utf-8"))
    selected = configured_indices(sweep, config)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    styles = {"SEED": ("#20578e", "o", "-"),
              "SEED-IV": ("#338c73", "s", "--"),
              "SEED-V": ("#bf5933", "^", "-.")}
    labels = ["(a) Batch size", "(b) Depth", r"(c) GRL $\lambda$", r"(d) Loss $\beta$"]
    for ax, parameter, label in zip(axes.flat, sweep["parameters"], labels):
        x = list(range(len(parameter["values"])))
        chosen = selected[parameter["key"]]
        ax.add_patch(Rectangle((chosen - 0.45, 65), 0.9, 40,
                               fill=False, linestyle="--", linewidth=1.2, edgecolor="black"))
        for dataset, (color, marker, linestyle) in styles.items():
            mean = parameter["results"][dataset]["mean"]
            std = parameter["results"][dataset]["std"]
            ax.fill_between(x, [a-b for a,b in zip(mean,std)],
                            [a+b for a,b in zip(mean,std)], color=color, alpha=0.07)
            ax.plot(x, mean, color=color, marker=marker, linestyle=linestyle,
                    markerfacecolor="white", linewidth=1.9, markersize=6, label=dataset)
            best = max(range(len(mean)), key=mean.__getitem__)
            ax.plot(best, mean[best], marker="*", color=color, markersize=13)
            ax.annotate(f"{mean[best]:.2f}", (best, mean[best]), xytext=(0,7),
                        textcoords="offset points", ha="center", color=color, fontsize=10)
        ax.set_xticks(x, [f"{value:g}" for value in parameter["values"]])
        ax.set_xlim(-0.3, len(x)-0.7)
        ax.set_ylim(65, 105)
        ax.set_yticks([70,80,90,100])
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel(label, fontsize=14)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Hyperparameter sensitivity", y=0.98, fontsize=14)
    fig.text(0.5, 0.01, "Dashed boxes: reference configuration. Stars: sweep maxima.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0,0.04,1,0.93), h_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)
    metadata = {"data_version": sweep["version"], "config_version": config["version"],
                "hash_format": "canonical-json-sort-keys",
                "data_sha256": hashlib.sha256(json.dumps(sweep, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                "selected_values": {p["key"]: p["values"][selected[p["key"]]]
                                    for p in sweep["parameters"]}}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
