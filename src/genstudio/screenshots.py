"""
Screenshot utilities for GenStudio plots with state change support
"""

import genstudio.layout as layout
import json
import time
import subprocess  # Added import for subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
from genstudio.util import WIDGET_URL, CSS_URL
from genstudio.chrome_devtools import ChromeContext


def update_state(chrome, state_updates, debug=False):
    if not isinstance(state_updates, list):
        raise AssertionError("state_updates must be a list")
    result = chrome.evaluate(
        f"""
                (async function() {{
                    try {{
                        window.last$state.update(...{json.dumps(state_updates)});
                        return window.genStudioReadyState.whenReady()
                    }} catch (e) {{
                        console.error('State update failed:', e);
                        return 'error: ' + e.message;
                    }}
                }})()
            """,
        await_promise=True,
    )
    if debug:
        print("State update result:", result)
    return result


def load_genstudio_html(chrome):
    if not chrome.evaluate("typeof window.genStudioRenderData === 'function'"):
        files = {}
        # Handle script content based on whether WIDGET_URL is a CDN URL or local file
        if isinstance(WIDGET_URL, str):  # CDN URL
            script_tag = f'<script type="module" src="{WIDGET_URL}"></script>'
            files = {}
        else:  # Local file
            script_tag = '<script type="module" src="studio.js"></script>'
            with open(WIDGET_URL, "r") as file:
                files["studio.js"] = file.read()
            files["studio.js"] = layout.get_script_source()
        if isinstance(CSS_URL, str):
            style_tag = f'<style>@import "{CSS_URL}";</style>'
        else:
            style_tag = '<style>@import "studio.css";</style>'
            with open(CSS_URL, "r") as file:
                files["studio.css"] = file.read()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>GenStudio</title>
            {style_tag}
            {script_tag}
        </head>
        <body>
            <div id="GenStudioView"></div>
        </body>
        </html>
        """

        chrome.set_content(html, files=files)


def measure_size(chrome):
    dimensions = chrome.evaluate("""
            (function() {
                const container = document.querySelector('.genstudio-container');
                if (!container) return null;
                const rect = container.getBoundingClientRect();
                return {
                    width: Math.ceil(rect.width),
                    height: Math.ceil(rect.height)
                };
            })()
        """)
    print("setting dimensions:", dimensions)
    if dimensions is not None:
        chrome.set_size(dimensions["width"], dimensions["height"])


def load_plot(chrome, plot, measure=True):
    load_genstudio_html(chrome)

    buffers = []
    data = layout.to_json_with_initialState(plot, buffers=buffers)

    chrome.evaluate(
        f"""
         (async () => {{
           genStudioRenderData('GenStudioView', {json.dumps(data)}, {layout.encode_buffers(buffers)});
           await window.genStudioReadyState.whenReady();
         }})()
         """,
        await_promise=True,
    )
    if measure:
        measure_size(chrome)


def take_screenshot(
    plot,
    output_path: Union[str, Path],
    state_update: Optional[Dict] = None,
    width: int = 400,
    height: int | None = None,
    debug: bool = False,
) -> Union[Path, bytes]:
    """
    Take a screenshot of a plot, optionally with a state update

    Args:
        plot: The GenStudio plot widget
        output_path: Path to save the screenshot
        state_update: Optional state update to apply before screenshot
        debug: Whether to print debug information

    Returns:
        Path to saved screenshot if output_path provided, otherwise raw bytes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with ChromeContext(width=width, height=height) as chrome:
        load_plot(chrome, plot)

        # Apply state update if provided
        if state_update:
            if not isinstance(state_update, dict):
                raise ValueError("State update must be a dictionary")
            update_state(chrome, [state_update], debug=debug)

        # Take and save screenshot
        return chrome.screenshot(output_path)


def take_screenshot_sequence(
    plot,
    state_updates: List[Dict],
    output_dir: Union[str, Path] = "./scratch/screenshots",
    filenames: Optional[List[str]] = None,
    filename_base: Optional[str] = "screenshot",
    width: int = 800,
    height: int | None = None,
    debug: bool = False,
) -> List[Path]:
    """
    Take a sequence of screenshots with state updates

    Args:
        plot: The GenStudio plot widget
        state_updates: List of state updates to apply
        output_dir: Directory to save screenshots
        filenames: Optional list of filenames for each screenshot. Must match length of state_updates
        filename_base: Base name for auto-generating filenames if filenames not provided.
                      Will generate names like "screenshot_0.png", "screenshot_1.png", etc.
        debug: Whether to print debug information

    Returns:
        List of paths to saved screenshots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate or validate filenames
    if filenames:
        if len(filenames) != len(state_updates):
            raise ValueError(
                f"Number of filenames ({len(filenames)}) must match number of state updates ({len(state_updates)})"
            )
    else:
        filenames = [f"{filename_base}_{i}.png" for i in range(len(state_updates))]

    output_paths = [output_dir / filename for filename in filenames]
    screenshots_taken = []

    with ChromeContext(width=width, height=height) as chrome:
        try:
            load_plot(chrome, plot)

            # Apply each state update and take screenshots
            for i, state_update in enumerate(state_updates):
                if not isinstance(state_update, dict):
                    raise ValueError(f"State update {i} must be a dictionary")
                update_state(chrome, [state_update])
                path = chrome.screenshot(output_paths[i])
                screenshots_taken.append(path)
                if debug:
                    print(f"Screenshot {i} taken: {path}")

            return screenshots_taken

        except Exception as e:
            if debug:
                import traceback

                traceback.print_exc()
            raise RuntimeError(f"Screenshot sequence failed: {e}")


def video(
    plot,
    state_updates: list,
    filename: Union[str, Path],
    fps: int = 24,
    width: int = 400,
    height: int | None = None,
    scale: float = 2.0,
    debug: bool = False,
) -> Path:
    print(f"Recording {len(state_updates)} frames...")
    start_time = time.time()
    """
    Capture a series of states from a plot as a movie, using the specified frame rate.
    The movie is generated without saving intermediate images to disk by piping PNG frames
    directly to ffmpeg.

    Args:
        plot: The GenStudio plot widget
        state_updates: List of state update dictionaries to apply sequentially
        output_path: Path where the resulting video will be saved
        frame_rate: Frame rate (frames per second) for the video
        debug: Whether to print debug information

    Returns:
        Path to the saved video file
    """
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)

    # Set up ffmpeg command to accept PNG images from a pipe and encode to MP4
    ffmpeg_cmd = (
        f"ffmpeg {'-v error' if not debug else ''} -y -f image2pipe -vcodec png -r {fps} -i - "
        f"-an -c:v libx264 -pix_fmt rgb24 {str(filename)}"
    )
    if debug:
        print(f"Running ffmpeg command: {ffmpeg_cmd}")

    # Start ffmpeg process with stdin as a pipe
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)

    with ChromeContext(width=width, height=height, scale=scale) as chrome:
        load_plot(chrome, plot)

        # Capture frames for each state update
        for i, state_update in enumerate(state_updates):
            if not isinstance(state_update, dict):
                raise ValueError(f"State update {i} must be a dictionary")
            result = chrome.evaluate(f"""
                (function() {{
                    try {{
                        window.last$state.update({json.dumps(state_update)});
                        return 'success';
                    }} catch (e) {{
                        console.error('State update failed:', e);
                        return 'error: ' + e.message;
                    }}
                }})()
            """)
            if debug:
                print(f"State update {i} result: {result}")
            # Capture frame after update
            frame_bytes = chrome.screenshot(None)
            if proc.stdin:
                proc.stdin.write(frame_bytes)
                if debug:
                    print(f"Captured frame {i}")

    # Close ffmpeg stdin and wait for process to finish
    if proc.stdin:
        proc.stdin.close()
    proc.wait()

    elapsed_time = time.time() - start_time
    actual_fps = len(state_updates) / elapsed_time
    print(
        f"   ...video generation took {elapsed_time:.2f} seconds ({actual_fps:.1f} fps)"
    )

    return filename


if __name__ == "__main__":
    import genstudio.plot as Plot
    from genstudio.scene3d import Ellipsoid

    # Test single screenshot
    test_plot = Plot.initialState({"test": "hello"}) | [
        "div",
        {"style": {"padding": "20px"}},
        Plot.js("$state.test"),
    ]
    take_screenshot(test_plot, "./scratch/screenshots/test.png", debug=True)

    # Test screenshot sequence
    counter_plot = (
        Plot.initialState({"count": 1})
        | [
            "div.bg-yellow-200.p-4",
            {"onClick": Plot.js("(e) => $state.clicks = ($state.clicks || 0) + 1")},
            Plot.js("`Count: ${$state.count}`"),
        ]
        | Plot.dot({"length": Plot.js("$state.count")}, x=Plot.index, y=Plot.index)
        + {"height": 200}
        | Ellipsoid(
            Plot.js("""
                Array.from({length: $state.count}, (_, i) => {
                    const t = i * Math.PI / 10;
                    return [
                        Math.cos(t),
                        Math.sin(t),
                        i / $state.count
                    ];
                }).flat()
            """),
            radius=0.1,
            color=[1, 0, 0],  # Red color for all ellipsoids
        )
    )

    take_screenshot(counter_plot, "./scratch/screenshots/_single.png")

    take_screenshot_sequence(
        counter_plot,
        state_updates=[{"count": i} for i in [1, 10, 100]],
        filename_base="count",
        debug=True,
        width=2000,
    )
    video(
        counter_plot,
        state_updates=[{"count": i} for i in range(60)],  # 60 frames
        filename="./scratch/screenshots/counter.mp4",
        fps=12,
        debug=False,
    )
