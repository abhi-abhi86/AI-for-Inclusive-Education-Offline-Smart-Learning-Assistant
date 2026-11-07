#!/usr/bin/env python3
"""
Unified GUI Launcher for Offline Smart Learning Assistant.
Supports:
1. Modern Lightweight Web GUI (Default): Launches zero-dependency local web server at http://localhost:8000
2. Native Desktop GUI: Launches Tkinter native window (--desktop)
"""

import argparse
import sys
import threading
import time
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Offline Smart Learning Assistant GUI")
    parser.add_argument("--desktop", action="store_true", help="Launch native Tkinter desktop window")
    parser.add_argument("--port", type=int, default=8000, help="Port for the lightweight web GUI (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser on startup")

    args = parser.parse_args()

    if args.desktop:
        from tools.gui_desktop import run_desktop_gui
        print("Launching Native Desktop GUI (Tkinter)...")
        run_desktop_gui()
    else:
        from gui_server import run_gui_server
        url = f"http://localhost:{args.port}"

        if not args.no_browser:
            def open_browser():
                time.sleep(0.6)
                webbrowser.open(url)

            threading.Thread(target=open_browser, daemon=True).start()

        run_gui_server(port=args.port)


if __name__ == "__main__":
    main()
