"""
Screenshot utilities for GenStudio plots with state change support
"""

import genstudio.widget as widget
import json
import time
import subprocess  # Added import for subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
from genstudio.util import read_file
from genstudio.html import encode_buffers
from genstudio.env import WIDGET_URL, CSS_URL
from genstudio.chrome_devtools import ChromeContext


def update_state(chrome, state_updates):
    if chrome.debug:
        print("[screenshots.py] Updating state")
    if not isinstance(state_updates, list):
        raise AssertionError("state_updates must be a list")
    buffers = []
    state_data = widget.to_json(state_updates, buffers=buffers)

    result = chrome.evaluate(
        f"""
        (async function() {{
            try {{
                const updates = {json.dumps(state_data)}
                const buffers = {encode_buffers(buffers)}
                const result = window.genstudio.instances['{chrome.id}'].updateWithBuffers(updates, buffers);
                await window.genstudio.whenReady('{chrome.id}');
                return result;
            }} catch (e) {{
                console.error('State update failed:', e);
                return 'error: ' + e.message;
            }}
        }})()
    """,
        await_promise=True,
    )
    return result


def load_genstudio_html(chrome):
    if not chrome.evaluate("typeof window.genstudio === 'object'"):
        if chrome.debug:
            print("[screenshots.py] Load html")

        files = {}
        # Handle script content based on whether WIDGET_URL is a CDN URL or local file
        if isinstance(WIDGET_URL, str):  # CDN URL
            if chrome.debug:
                print(f"[screenshots.py] Using CDN script from: {WIDGET_URL}")
            script_tag = f'<script type="module" src="{WIDGET_URL}"></script>'
        else:  # Local file
            if chrome.debug:
                print(f"[screenshots.py] Loading local script from: {WIDGET_URL}")
            script_tag = '<script type="module" src="studio.js"></script>'
            files["studio.js"] = read_file(WIDGET_URL)

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
            <div id="studio"></div>
        </body>
        </html>
        """
        chrome.load_html(html, files=files)
    elif chrome.debug:
        print("GenStudio already loaded, skipping initialization")


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
    if chrome.debug:
        print(f"Measured container dimensions: {dimensions}")
    if dimensions is not None:
        chrome.set_size(dimensions["width"], dimensions["height"])


def load_plot(chrome, plot, measure=True):
    if chrome.debug:
        print("Loading plot into GenStudio")

    load_genstudio_html(chrome)

    data, buffers = widget.to_json_with_initialState(plot, buffers=[])

    if chrome.debug:
        print("Rendering plot data")
        print(f"Buffer count: {len(buffers)}")

    chrome.evaluate(
        f"""
         (async () => {{
           window.genstudio.renderData('studio', {json.dumps(data)}, {encode_buffers(buffers)}, '{chrome.id}');
           await window.genstudio.whenReady('{chrome.id}');
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
    if debug:
        print(f"Taking screenshot, saving to: {output_path}")
        print(f"Window size: {width}x{height}")

    output_path.parent.mkdir(exist_ok=True, parents=True)

    with ChromeContext(width=width, height=height, debug=debug) as chrome:
        load_plot(chrome, plot)
        update_state(chrome, [state_update or {}])
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
    if debug:
        print("Taking screenshot sequence")
        print(f"Output directory: {output_dir}")
        print(f"Number of updates: {len(state_updates)}")

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

    with ChromeContext(width=width, height=height, debug=debug) as chrome:
        try:
            load_plot(chrome, plot)

            for i, state_update in enumerate(state_updates):
                if debug:
                    print(f"Processing state update {i+1}/{len(state_updates)}")
                if not isinstance(state_update, dict):
                    raise ValueError(f"State update {i} must be a dictionary")
                update_state(chrome, [state_update])
                path = chrome.screenshot(output_paths[i])
                screenshots_taken.append(path)
                if debug:
                    print(f"Saved screenshot to: {path}")

            return screenshots_taken

        except Exception as e:
            if debug:
                import traceback

                print("Screenshot sequence failed:")
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
    filename = Path(filename)
    if debug:
        print(f"Recording video with {len(state_updates)} frames")

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
        f"-an -c:v libx264 -pix_fmt yuv420p {str(filename)}"
    )
    if debug:
        print(f"Running ffmpeg command: {ffmpeg_cmd}")

    # Start ffmpeg process with stdin as a pipe
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)

    with ChromeContext(width=width, height=height, scale=scale, debug=debug) as chrome:
        load_plot(chrome, plot)

        # Capture frames for each state update
        for i, state_update in enumerate(state_updates):
            result = update_state(chrome, [state_update])
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
    if debug:
        print(
            f"   ...video generation took {elapsed_time:.2f} seconds ({actual_fps:.1f} fps)"
        )

    return filename
