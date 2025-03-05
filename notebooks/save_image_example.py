# %% [markdown]
# ## Saving Plots as Images and Videos
#
# This notebook shows how to save plots as static images and videos.

# %%
import genstudio.plot as Plot
from genstudio.scene3d import Ellipsoid
from pathlib import Path

# Create output directory
output_dir = Path("scratch/export_examples")
output_dir.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ### Static Images
#
# Save a plot as a static image:

# %%
# Create and display a simple scatter plot
dots = Plot.dot([[1, 1], [2, 2], [3, 3]], r=10, fill="steelblue")
dots.save_image(str(output_dir / "scatter.png"), width=400)

# %% [markdown]
# ### Image Sequences
#
# Save a plot in different states as a sequence of images. Here's a plot that
# arranges points in a circle, with the number of points controlled by state:

# %%
# Create a plot with state
circle_plot = Plot.initialState({"count": 5}) | [
    "div",
    {"style": {"padding": "20px", "backgroundColor": "#f0f0f0"}},
    ["h3", Plot.js("`Points: ${$state.count}`")],
    Plot.dot(
        {"length": Plot.js("$state.count")},
        x=Plot.js("(d,i) => Math.cos(i * Math.PI * 2 / $state.count)"),
        y=Plot.js("(d,i) => Math.sin(i * Math.PI * 2 / $state.count)"),
        r=8,
        fill="steelblue",
    ),
]

# Display the initial state
circle_plot

# %% [markdown]
# Save images with different numbers of points:

# %%
# Save multiple states as separate images
paths = circle_plot.save_images(
    state_updates=[{"count": i} for i in [3, 6, 12, 24]],
    output_dir=output_dir,  # Convert Path to str
    filename_base="circle",
    width=500,
)

print("Created images:")
for path in paths:
    print(f"  {path}")

# %% [markdown]
# ### Videos
#
# Create a video by animating state transitions. This example shows animated
# points in 3D space:

# %%
# Create a 3D scene with animated points
animated_scene = Plot.initialState({"t": 0}) | Ellipsoid(
    Plot.js("""
            Array.from({length: 50}, (_, i) => {
                const angle = i * Math.PI * 2 / 50;
                const x = Math.cos(angle + $state.t);
                const y = Math.sin(angle + $state.t);
                const z = Math.sin($state.t * 2 + i * 0.1);
                return [x, y, z];
            }).flat()
        """),
    radius=0.1,
    color=Plot.js("""
            (d, i) => {
                const j = Math.floor(i / 3);
                return [
                    0.5 + 0.5 * Math.sin($state.t + j * 0.1),
                    0.5 + 0.5 * Math.cos($state.t + j * 0.2),
                    0.5 + 0.5 * Math.sin($state.t + j * 0.3)
                ];
            }
        """),
)

# Display the initial state
animated_scene

# %%
# Save as video if ffmpeg is available
import shutil

if shutil.which("ffmpeg"):
    video_path = animated_scene.save_video(
        state_updates=[{"t": i * 0.1} for i in range(60)],
        filename=str(output_dir / "points.mp4"),  # Convert Path to str
        fps=30,
        width=800,
        height=600,
    )
    print(f"Video saved to: {video_path}")
else:
    print("Note: Video creation requires ffmpeg to be installed")

# %% [markdown]
# ### PDF Export
#
# PDF export is supported via `.save_pdf(...)`, however note that 3d canvas elements will not render.

# %%
# Create and display a simple scatter plot
dots = Plot.dot([[1, 1], [2, 2], [3, 3]], r=10, fill="steelblue")
dots.save_pdf(str(output_dir / "scatter.pdf"), scale=2, width=400)

print(f"PDF saved to: {output_dir / 'scatter.pdf'}")
