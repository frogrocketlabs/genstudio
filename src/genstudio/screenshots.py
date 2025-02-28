"""
Screenshot utilities for GenStudio plots with state change support
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from genstudio.chrome_devtools import ChromeContext
from genstudio.layout import html_standalone


def take_screenshot(
    plot,
    output_path: Union[str, Path],
    state_update: Optional[Dict] = None,
    window_size: tuple[int, int] = (1200, 800),
    wait_time: float = 0.0,
    debug: bool = False,
) -> Union[Path, bytes]:
    """
    Take a screenshot of a plot, optionally with a state update

    Args:
        plot: The GenStudio plot widget
        output_path: Path to save the screenshot
        state_update: Optional state update to apply before screenshot
        window_size: (width, height) in pixels
        wait_time: Seconds to wait after state changes
        debug: Whether to print debug information

    Returns:
        Path to saved screenshot if output_path provided, otherwise raw bytes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with ChromeContext(window_size=window_size) as chrome:
        # Get and set the HTML content
        html_content = html_standalone(plot.for_json())
        chrome.set_content(html_content)
        time.sleep(wait_time)  # Wait for initial render

        # Apply state update if provided
        if state_update:
            if not isinstance(state_update, dict):
                raise ValueError("State update must be a dictionary")

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
                print("State update result:", result)

            time.sleep(wait_time)  # Wait for state update to apply

        # Take and save screenshot
        return chrome.screenshot(output_path)


def take_screenshot_sequence(
    plot,
    state_updates: List[Dict],
    output_dir: Union[str, Path] = "./scratch/screenshots",
    filenames: Optional[List[str]] = None,
    window_size: tuple[int, int] = (1200, 800),
    wait_time: float = 0.1,
    debug: bool = False,
) -> List[Path]:
    """
    Take a sequence of screenshots with state updates

    Args:
        plot: The GenStudio plot widget
        state_updates: List of state updates to apply
        output_dir: Directory to save screenshots
        filenames: Optional list of filenames (default: screenshot_0.png, etc.)
        window_size: (width, height) in pixels
        wait_time: Seconds to wait after state changes
        debug: Whether to print debug information

    Returns:
        List of paths to saved screenshots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Prepare filenames (need one more than state updates for initial state)
    if filenames is None:
        filenames = [f"screenshot_{i}.png" for i in range(len(state_updates) + 1)]
    elif len(filenames) != len(state_updates) + 1:
        raise ValueError(
            f"Number of filenames ({len(filenames)}) must be one more than "
            f"number of state updates ({len(state_updates)})"
        )

    output_paths = [output_dir / filename for filename in filenames]
    screenshots_taken = []

    with ChromeContext(window_size=window_size) as chrome:
        try:
            # Get and set the HTML content
            html_content = html_standalone(plot.for_json())
            chrome.set_content(html_content)
            time.sleep(wait_time)  # Wait for initial render

            # Take initial screenshot
            initial_path = chrome.screenshot(output_paths[0])
            screenshots_taken.append(initial_path)
            if debug:
                print(f"Initial screenshot taken: {initial_path}")

            # Apply each state update and take screenshots
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
                    print(f"State update {i} result:", result)

                time.sleep(wait_time)  # Wait for state update to apply

                path = chrome.screenshot(output_paths[i + 1])
                screenshots_taken.append(path)
                if debug:
                    print(f"Screenshot {i+1} taken: {path}")

            return screenshots_taken

        except Exception as e:
            if debug:
                import traceback

                traceback.print_exc()
            raise RuntimeError(f"Screenshot sequence failed: {e}")


if __name__ == "__main__":
    import genstudio.plot as Plot

    # Test single screenshot
    test_plot = Plot.initialState({"test": "hello"}) | [
        "div",
        {"style": {"padding": "20px"}},
        Plot.js("$state.test"),
    ]
    take_screenshot(test_plot, "test.png", debug=True)

    # Test screenshot sequence
    counter_plot = Plot.initialState({"count": 0}) | [
        "div.bg-yellow-200.p-4",
        {"onClick": Plot.js("(e) => $state.clicks = ($state.clicks || 0) + 1")},
        Plot.js("`Count: ${$state.count}`"),
    ]

    take_screenshot_sequence(
        counter_plot,
        state_updates=[{"count": i} for i in [1, 10, 100]],
        filenames=["count_0.png", "count_1.png", "count_10.png", "count_100.png"],
        debug=True,
    )
