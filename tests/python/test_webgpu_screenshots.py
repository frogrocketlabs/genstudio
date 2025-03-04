"""
Tests for WebGPU screenshot functionality in GenStudio
"""

import os
import shutil
from pathlib import Path
from genstudio.screenshots import take_screenshot, take_screenshot_sequence, video
import genstudio.plot as Plot
from genstudio.scene3d import Ellipsoid
from genstudio.chrome_devtools import ChromeContext

# Create an artifacts directory for screenshots
ARTIFACTS_DIR = Path(__file__).parent / "screenshot-artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)


def test_basic_screenshot():
    """Test basic screenshot functionality"""
    test_plot = Plot.initialState({"test": "hello"}) | [
        "div",
        {"style": {"padding": "20px"}},
        Plot.js("$state.test"),
    ]

    screenshot_path = ARTIFACTS_DIR / "test.png"
    take_screenshot(test_plot, screenshot_path, debug=True)

    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 0


def test_counter_plot():
    """Test more complex plot with state updates"""
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

    # Test single screenshot
    single_path = ARTIFACTS_DIR / "_single.png"
    take_screenshot(counter_plot, single_path, debug=True)
    assert single_path.exists()

    # Test screenshot sequence
    paths = take_screenshot_sequence(
        counter_plot,
        state_updates=[{"count": i} for i in [1, 10, 100]],
        output_dir=ARTIFACTS_DIR,
        filename_base="count",
        debug=True,
        width=2000,
    )
    for path in paths:
        assert path.exists()

    # Test video generation
    if not os.environ.get("CI") or shutil.which("ffmpeg"):
        video_path = ARTIFACTS_DIR / "counter.mp4"
        video(
            counter_plot,
            state_updates=[{"count": i} for i in range(60)],  # 60 frames
            filename=video_path,
            fps=12,
            debug=True,
        )
        assert video_path.exists()
        assert video_path.stat().st_size > 0


if __name__ == "__main__":
    test_basic_screenshot()

    with ChromeContext(width=400, debug=True) as chrome:
        support = chrome.check_webgpu_support()
        if support["supported"]:
            print("WebGPU is fully functional")
            print(f"Using GPU: {support['adapter']['name']}")
        else:
            print(f"WebGPU not available: {support['reason']}")

    test_counter_plot()
