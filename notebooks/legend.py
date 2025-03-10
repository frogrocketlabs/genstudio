import genstudio.plot as Plot
import numpy as np

x = np.linspace(0, 10, 100)
fps_plot = (
    Plot.line([[x, y] for x, y in zip(x, np.sin(x))], stroke=Plot.constantly("Gen3D"))
    + Plot.line(
        [[x, y] for x, y in zip(x, np.sin(x + 0.5))],
        stroke=Plot.constantly("Gaussian Splatting"),
    )
    + Plot.line(
        [[x, y] for x, y in zip(x, np.cos(x + 0.5))],
        stroke=Plot.constantly("FoundationPose"),
    )
    + Plot.colorLegend()
    + {
        "x": {"label": "Frames per second"},
        "y": {"label": "Gaussian count"},
        "height": 160,
    }
)

fps_plot

Plot.legend(fps_plot, "color")
