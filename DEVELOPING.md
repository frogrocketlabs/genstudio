# Developer's Guide

This guide covers common development tasks in the GenStudio codebase.

## CI/CD

### Screenshots and Videos

GenStudio uses a headless Chrome browser to generate screenshots and videos of visualizations. This is used for:

- Testing visualizations
- Documentation examples
- State transition animations
- 3D scene captures

#### Implementation

Two main modules handle this functionality:

- `genstudio.screenshots`: Screenshot and video generation API
- `genstudio.chrome_devtools`: Chrome DevTools Protocol client

Available operations:
- Single screenshots
- Multiple screenshots with state updates
- Video generation (requires ffmpeg)
- Custom dimensions and scaling

#### Example Usage

```python
from genstudio.screenshots import take_screenshot, take_screenshot_sequence, video

# Take a screenshot
take_screenshot(plot, "output.png", width=800)

# Take multiple screenshots with different states
take_screenshot_sequence(
    plot,
    state_updates=[{"param": i} for i in range(5)],
    output_dir="./screenshots"
)

# Create a video
video(
    plot,
    state_updates=[{"t": i/30} for i in range(60)],
    filename="animation.mp4",
    fps=30
)
```

### WebGPU Testing

The GitHub Actions workflow tests WebGPU rendering by taking screenshots of 3D scenes. This verifies that WebGPU works correctly across environments.

The workflow:
1. Runs Chrome with WebGPU in headless mode
2. Creates and captures 3D scenes
3. Saves screenshots as artifacts

Tests in `tests/python/test_webgpu_screenshots.py` check:
- WebGPU availability
- 3D primitive rendering
- State updates
- Animations

Run tests locally:

```bash
poetry run pytest tests/python/test_webgpu_screenshots.py -v
```

### Jupyter notes

A typical and recommended workflow is to use genstudio with VS Code's Python Interactive Window. With the VS Code jupyter extension installed, one can use ordinary `.py` files with `# %%` markers to separate cells, then run the `Jupyter: Run Current Cell` command. Results, including plots, will be rendered with VS Code.

Of course, one can also use genstudio from within Jupyter Labs and Colab.

If jupyter has trouble finding a kernel to evaluate from, you can install one (using poetry) via:

```bash
poetry run python -m ipykernel install --user --name genstudio
```

### Pre-commit Hooks

Pre-commit hooks ensure code consistency. They run automatically on each commit to format Python code and perform other checks.

Setup:

1. Install pre-commit:
```bash
pipx install pre-commit
```

2. Install hooks:
```bash
pre-commit install
```

Run hooks manually:
```bash
pre-commit run --all-files
```

Hooks are configured in `.pre-commit-config.yaml`.
