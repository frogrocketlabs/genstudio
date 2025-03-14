from genstudio.scene3d import Ellipsoid, deco
import numpy as np
from genstudio.plot import js

Ellipsoid(
    # fill_mode="MajorWireframe",
    centers=np.array([[1, 0, 0], [1, 1, 0], [1, 0.5, 1]]),  # Offset by 1 in x direction
    color=[0, 1, 0],
    half_sizes=[0.4, 0.4, 0.4],
    alpha=0.8,
    quaternions=np.array(
        [[0.866, 0, 0.5, 0], [0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5]]
    ),
    onHover=js("(i) => $state.update({hover_axes1: typeof i === 'number' ? [i] : []})"),
    decorations=[
        deco([2], color=[1, 0, 0]),
        deco(
            js("$state.hover_axes1"),
            color=[1, 1, 0],
        ),
        deco([0], scale=1.5),
        deco([1], scale=0.5),
    ],
) & Ellipsoid(
    fill_mode="MajorWireframe",
    centers=np.array([[1, 0, 0], [1, 1, 0], [1, 0.5, 1]]),  # Offset by 1 in x direction
    color=[0, 1, 0],
    half_sizes=[0.4, 0.4, 0.4],
    alpha=0.8,
    quaternions=np.array(
        [[0.866, 0, 0.5, 0], [0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5]]
    ),
    onHover=js("(i) => $state.update({hover_axes1: typeof i === 'number' ? [i] : []})"),
    decorations=[
        deco([2], color=[1, 0, 0]),
        deco(
            js("$state.hover_axes1"),
            color=[1, 1, 0],
        ),
        deco([0], scale=1.5),
        deco([1], scale=0.5),
    ],
)


(
    Ellipsoid(
        fill_mode="MajorWireframe",
        centers=np.array(
            [[2, 0, 0], [2, 1, 0], [2, 0.5, 1]]
        ),  # Offset by 2 in x direction
        color=[1, 0, 1],
        half_sizes=[0.8, 0.1, 0.1],  # Very squished ellipsoids
        quaternions=np.array(
            [[0.866, 0, 0.5, 0], [0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5]]
        ),
        onHover=js(
            "(i) => $state.update({hover_axes2: typeof i === 'number' ? [i] : []})"
        ),
        decorations=[
            deco([2], color=[1, 0, 0]),
            deco(
                js("$state.hover_axes2"),
                color=[1, 1, 0],
            ),
            deco([0], scale=1.5),
            deco([1], scale=0.5),
        ],
    )
    + Ellipsoid(
        centers=np.array(
            [[2, 0, 0], [2, 1, 0], [2, 0.5, 1]]
        ),  # Offset by 2 in x direction
        color=[1, 0, 1],
        half_sizes=[0.8, 0.1, 0.1],  # Very squished ellipsoids
        alpha=0.1,
        quaternions=np.array(
            [[0.866, 0, 0.5, 0], [0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5]]
        ),
        onHover=js(
            "(i) => $state.update({hover_axes2: typeof i === 'number' ? [i] : []})"
        ),
        decorations=[
            deco([2], color=[1, 0, 0]),
            deco(
                js("$state.hover_axes2"),
                color=[1, 1, 0],
            ),
            deco([0], scale=1.5),
            deco([1], scale=0.5),
        ],
    )
    + {
        "defaultCamera": {
            "position": [6.261824, 2.239899, 1.435801],
            "target": [0.000000, 0.000000, 0.000000],
            "up": [0.000000, 1.000000, 0.000000],
            "fov": 45,
        }
    }
)
