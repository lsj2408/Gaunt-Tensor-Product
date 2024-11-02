import argparse
from typing import Tuple

import ase.data
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import style_dict

plt.rcParams.update({"font.size": 6})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", action="append", required=True)
    return parser.parse_args()


def parse_configs(name_path_tuple: str) -> Tuple[str, pd.DataFrame]:
    name, *paths = name_path_tuple.split(",")
    frames = [
        {"energy": atoms.info["energy"], "forces": atoms.info["forces"],}
        for path in paths
        for atoms in ase.io.read(path, format="extxyz", index=":")
    ]

    df = (
        pd.DataFrame(frames)
        .groupby(["d1", "d2"])
        .aggregate(
            mean_energy=pd.NamedAgg(column="energy", aggfunc="mean"),
            std_energy=pd.NamedAgg(column="energy", aggfunc="std"),
        )
        .reset_index()
    )

    return name, df


def main():
    args = parse_args()
    training_atoms = ase.io.read(args.training, format="extxyz", index=":")
    dihedral_predictions = [
        parse_dihedral_configs(path) for path in args.dihedral_configs
    ]
    transfer_predictions = [
        parse_transfer_configs(path) for path in args.transfer_configs
    ]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(6.0, 2.1),
        constrained_layout=True,
        sharex="col",
        gridspec_kw={"height_ratios": (5, 1)},
    )

    # Dihedral curve
    ax = axes[0][0]
    ref_energy = np.min(dihedral_predictions[0][1]["mean_energy"])

    for index, (name, df) in enumerate(dihedral_predictions):
        ax.plot(
            df["dihedral"],
            (df["mean_energy"] - ref_energy),
            zorder=2 * index + 1,
            **style_dict[name],
        )
        ax.fill_between(
            x=df["dihedral"],
            y1=df["mean_energy"] - ref_energy - df["std_energy"],
            y2=df["mean_energy"] - ref_energy + df["std_energy"],
            alpha=0.3,
            zorder=2 * index,
            **style_dict[name],
        )

    ax.set_ylabel(r"$\Delta E$ [meV]")
    ax.legend().set_zorder(2 * len(dihedral_predictions))

    # Dihedral Histogram
    ax = axes[1][0]
    train_dihedrals = []
    for atoms in training_atoms:
        if atoms.get_dihedral(0, 1, 2, 3) < 180:
            train_dihedrals.append(atoms.get_dihedral(0, 1, 2, 3))
        else:
            train_dihedrals.append(360 - atoms.get_dihedral(0, 1, 2, 3))

    ax.hist(
        train_dihedrals, bins=np.arange(0, 185, 5), color="black", label="Training data"
    )
    ax.set_xlabel("Dihedral Angle [°]")
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_ylabel("Count")
    ax.legend()

    # H transfer
    ax = axes[0][1]
    ref_energy = np.min(transfer_predictions[0][1]["mean_energy"])
    for index, (name, df) in enumerate(transfer_predictions):
        ax.plot(
            df["d1"],
            df["mean_energy"] - ref_energy,
            zorder=2 * index + 1,
            **style_dict[name],
        )
        ax.fill_between(
            x=df["d1"],
            y1=df["mean_energy"] - ref_energy - df["std_energy"],
            y2=df["mean_energy"] - ref_energy + df["std_energy"],
            alpha=0.3,
            zorder=2 * index,
            **style_dict[name],
        )

    # H transfer histogram
    ax = axes[1][1]
    step_size = 0.05
    d_range = np.arange(1.05, 1.50, step=step_size)
    d1s = np.array([atoms.get_distance(3, 11) for atoms in training_atoms])
    d2s = np.array([atoms.get_distance(5, 11) for atoms in training_atoms])
    train_hist = np.histogram2d(d1s, d2s, bins=d_range)[0]
    sym_hist = train_hist + train_hist.transpose()

    ref_df = transfer_predictions[0][1]
    path_hist = np.histogram2d(ref_df["d1"], ref_df["d2"], bins=d_range)[0]
    hist = sym_hist * path_hist

    ax.bar(
        x=0.5 * (d_range[:-1] + d_range[1:]),
        height=np.sum(hist, axis=1),
        width=step_size,
        color="black",
        label="Training data",
    )

    ax.set_xlabel("Distance [Å]")
    ax.set_ylabel("Count")

    fig.savefig("acac.pdf")


if __name__ == "__main__":
    main()
