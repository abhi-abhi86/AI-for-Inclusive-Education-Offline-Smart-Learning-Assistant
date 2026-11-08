#!/usr/bin/env python3
"""
Unified GUI Launcher for Offline Smart Learning Assistant.
Supports:
1. Modern Lightweight Web GUI (Default): Launches zero-dependency local web server at http://localhost:8000
2. Native Desktop GUI: Launches Tkinter native window (--desktop)
"""

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser


def _use_project_python() -> None:
    """Restart with the project's dependency environment when available."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        python_path = os.path.join(project_root, "venv", "Scripts", "python.exe")
    else:
        python_path = os.path.join(project_root, "venv", "bin", "python")

    if (
        os.path.isfile(python_path)
        and sys.prefix == sys.base_prefix
    ):
        os.execv(python_path, [python_path, os.path.abspath(__file__), *sys.argv[1:]])


def main():
    _use_project_python()
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

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            port_in_use = probe.connect_ex(("127.0.0.1", args.port)) == 0
        if port_in_use:
            print(
                f"GUI port {args.port} is already in use. "
                f"Opening the existing server at {url}."
            )
            if not args.no_browser:
                webbrowser.open(url)
            return

        if not args.no_browser:
            def open_browser():
                time.sleep(0.6)
                webbrowser.open(url)

            threading.Thread(target=open_browser, daemon=True).start()

        run_gui_server(port=args.port)


if __name__ == "__main__":
    main()
