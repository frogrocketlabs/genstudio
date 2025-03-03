"""
Simple Chrome DevTools Protocol client for HTML content manipulation and screenshots
"""

import os
import time
import json
import base64
import shutil
import subprocess
import urllib.request
import websocket
import http.server
import socketserver
import threading
import tempfile
from functools import partial

DEBUG = False


def find_chrome():
    """Find Chrome executable on the system"""
    possible_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
        "/usr/bin/google-chrome",  # Linux
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # Windows
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    # Check PATH first
    for cmd in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        chrome_path = shutil.which(cmd)
        if chrome_path:
            return chrome_path

    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find Chrome. Please install Chrome.")


class ChromeContext:
    """Manages a Chrome instance and provides methods for content manipulation and screenshots"""

    def __init__(self, port=9222, width=400, height=None, scale=1.0):
        self.port = port
        self.width = width
        self.height = height
        self.scale = scale
        self.chrome_process = None
        self.ws = None
        self.cmd_id = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def set_size(self, width=None, height=None, scale=None):
        self.width = width or self.width
        self.height = height or self.height or self.width
        if scale:
            self.scale = scale
        self._send_command(
            "Browser.setWindowBounds",
            {
                "windowId": self._send_command("Browser.getWindowForTarget")[
                    "windowId"
                ],
                "bounds": {"width": self.width, "height": self.height},
            },
        )
        self._send_command(
            "Page.setDeviceMetricsOverride",
            {
                "width": self.width,
                "height": self.height,
                "deviceScaleFactor": self.scale,
                "mobile": False,
            },
        )

    def start(self):
        """Start Chrome and connect to DevTools Protocol"""
        if self.chrome_process:
            self.set_size()
            return  # Already started

        chrome_path = find_chrome()
        chrome_cmd = [
            chrome_path,
            f"--remote-debugging-port={self.port}",
            "--remote-allow-origins=*",
            "--disable-search-engine-choice-screen",
            "--ash-no-nudges",
            "--no-first-run",
            "--disable-features=Translate",
            "--no-default-browser-check",
            "--headless=new" if not DEBUG else "",
            "--hide-scrollbars",
            f"--window-size={self.width},{self.height or self.width}",
            "--app=data:,",
            # maybe need these for linux?
            # "--enable-features=Vulkan,UseSkiaRenderer,WebGPU",
            # "--enable-unsafe-webgpu",
            # "--use-vulkan=native"
        ]

        self.chrome_process = subprocess.Popen(
            chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for Chrome to start by polling
        start_time = time.time()
        while True:
            try:
                response = urllib.request.urlopen(f"http://localhost:{self.port}/json")
                targets = json.loads(response.read())
                page_target = next(
                    (
                        t
                        for t in targets
                        if t["type"] == "page" and t["url"] == "data:,"
                    ),
                    None,
                )
                if page_target:
                    break
            except Exception:
                pass

            if time.time() - start_time > 10:  # Timeout after 10 seconds
                raise RuntimeError("Chrome did not start in time")
        # Connect to the page target
        self.ws = websocket.create_connection(page_target["webSocketDebuggerUrl"])
        # Enable required domains
        self._send_command("Page.enable")
        self._send_command("Runtime.enable")

    def stop(self):
        """Stop Chrome and clean up"""
        if self.ws:
            self.ws.close()
            self.ws = None

        if self.chrome_process and not DEBUG:
            self.chrome_process.terminate()
            try:
                self.chrome_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.chrome_process.kill()
            self.chrome_process = None

    def _send_command(self, method, params=None):
        """Send a command to Chrome and wait for the response"""
        if not self.ws:
            raise RuntimeError("Not connected to Chrome")

        self.cmd_id += 1
        message = {"id": self.cmd_id, "method": method, "params": params or {}}

        self.ws.send(json.dumps(message))

        # Wait for response with matching id
        while True:
            response = json.loads(self.ws.recv())
            if "id" in response and response["id"] == self.cmd_id:
                if "error" in response:
                    raise RuntimeError(
                        f"Chrome DevTools command failed: {response['error']}"
                    )

                return response.get("result", {})

    def set_content(self, html, files=None):
        """Serve HTML content and optional files over localhost and load it in the page

        Args:
            html: HTML content for index.html
            files: Optional dict of {filename: content} to serve alongside index.html
        """
        # Ensure viewport size is set correctly
        self.set_size()

        # Create a temporary directory and write the HTML content to an index file
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Write index.html
            file_path = os.path.join(tmp_dir, "index.html")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html)

            # Write any additional files
            if files:
                for filename, content in files.items():
                    file_path = os.path.join(tmp_dir, filename)
                    # Create subdirectories if needed
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

            # Set up a simple HTTP server to serve the temporary directory
            handler = partial(http.server.SimpleHTTPRequestHandler, directory=tmp_dir)
            with socketserver.TCPServer(("localhost", 0), handler) as httpd:
                port = httpd.server_address[1]
                # Start the server in a background thread
                server_thread = threading.Thread(
                    target=httpd.serve_forever, kwargs={"poll_interval": 0.1}
                )
                server_thread.daemon = True
                server_thread.start()

                # Navigate to the served page in the same tab
                url = f"http://localhost:{port}/index.html"
                self._send_command("Page.navigate", {"url": url})

                # Wait for page load
                while True:
                    if not self.ws:
                        raise RuntimeError("WebSocket connection lost")
                    response = json.loads(self.ws.recv())
                    if response.get("method") == "Page.loadEventFired":
                        break

                # Shutdown the HTTP server
                httpd.shutdown()
                server_thread.join()

    def evaluate(self, expression, return_by_value=True, await_promise=False):
        """Evaluate JavaScript code in the page context

        Args:
            expression: JavaScript expression to evaluate
            return_by_value: Whether to return the result by value
            wait_for_ready: Whether to wait for readyState after evaluation
                          (useful for state updates that trigger async rendering)
        """
        result = self._send_command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": await_promise,
            },
        )

        return result.get("result", {}).get("value")

    def screenshot(self, path=None):
        """Take a screenshot of the page, automatically sizing to content height"""
        result = self._send_command(
            "Page.captureScreenshot",
            {
                "format": "png",
                "captureBeyondViewport": True,
                "clip": {
                    "x": 0,
                    "y": 0,
                    "width": self.width,
                    "height": self.height,
                    "scale": self.scale,
                },
            },
        )

        if not result or "data" not in result:
            raise RuntimeError("Failed to capture screenshot")

        image_data = base64.b64decode(result["data"])
        if path:
            with open(path, "wb") as f:
                f.write(image_data)
            return path
        return image_data

    def check_webgpu_support(self):
        """Check if WebGPU is available in the browser"""
        result = self.evaluate("""
            (function() {
                if (!navigator.gpu) {
                    return { supported: false, reason: 'navigator.gpu is not available' };
                }

                try {
                    return {
                        supported: true,
                        info: {
                            gpu: !!navigator.gpu,
                            requestAdapter: typeof navigator.gpu.requestAdapter === 'function'
                        }
                    };
                } catch (e) {
                    return { supported: false, reason: e.toString() };
                }
            })()
        """)

        return result


def main():
    """Example usage"""
    html = """
    <html>
    <head>
        <style>
            body { margin: 0; }
            div { width: 100vw; height: 100vh; background: red; }
        </style>
    </head>
    <body><div></div></body>
    </html>
    """

    with ChromeContext(width=400, height=600) as chrome:
        # Load content served via localhost
        chrome.set_content(html)
        # Evaluate some JavaScript
        result = chrome.evaluate('document.body.style.background = "green"; "changed!"')
        print("Eval result:", result)

        # Check WebGPU support
        webgpu_status = chrome.check_webgpu_support()
        print("WebGPU Support:", webgpu_status)

        # Take a screenshot
        screenshot_path = chrome.screenshot("./scratch/screenshots/webgpu_test.png")
        print(f"Screenshot saved to: {screenshot_path}")

        # Wait a bit to see the window if DEBUG is True
        if DEBUG:
            print("Press Ctrl+C to exit...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
