from typing import Dict, Any

colors = [
    "#1f77b4",  # muted blue
    "#d62728",  # brick red
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]

dashed = (0, (5, 1))
dotted = (0, (1, 1))

style_dict: Dict[str, Dict[str, Any]] = {
    "dft": {"color": "black", "label": "DFT", "linestyle": dotted, "linewidth": 2.5,},
    "nequip": {"color": colors[1], "label": "NequIP", "linewidth": 1.5,},
    "nequip-linear": {
        "color": colors[3],
        "label": "NequIP Linear",
        "linestyle": dashed,
        "linewidth": 1.5,
    },
    "nequip-shifted": {"color": colors[4], "label": "NequIP Shift", "linewidth": 1.5,},
    "nequip-bo": {"color": colors[5], "label": "NequIP BO", "linewidth": 1.5,},
    "nequip-e0": {"color": colors[3], "label": "NequIP E0", "linewidth": 1.5,},
    "nequip-ssh": {
        "color": colors[4],
        "label": "NequIP SSH",
        "linestyle": dashed,
        "linewidth": 1.5,
    },
    "botnet": {"color": colors[0], "label": "BOTNet", "linewidth": 1.5,},
    "botnet-e0": {"color": colors[0], "label": "BOTNet E0", "linewidth": 1.5,},
    "botnet-ssh": {
        "color": colors[8],
        "label": "BOTNet-SSH",
        "linestyle": dashed,
        "linewidth": 1.5,
    },
    "ace": {"color": colors[2], "label": "linACE", "linewidth": 1.5,},
    "multiACE-ssh": {
        "color": colors[1],
        "label": "multiACE",
        "linestyle": dashed,
        "linewidth": 1.5,
    },
}
