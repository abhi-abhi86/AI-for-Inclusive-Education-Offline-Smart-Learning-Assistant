#!/usr/bin/env python3
"""Run a local command while displaying a responsive progress window."""

import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import re
from tkinter import ttk
from typing import List, Optional


class ProgressWindow:
    def __init__(self, command: List[str], title: str) -> None:
        self.command = command
        self.title_text = title
        self.training_phase = False
        self.messages: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.process: Optional[subprocess.Popen[str]] = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("680x390")
        self.root.minsize(520, 300)
        self.root.protocol("WM_DELETE_WINDOW", self._close_window)

        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=title, font=("TkDefaultFont", 15, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame,
            text="The operation is running locally. You can continue when it completes.",
        ).pack(anchor="w", pady=(6, 14))

        self.status = ttk.Label(frame, text="Starting...")
        self.status.pack(anchor="w")
        self.percent = ttk.Label(frame, text="0%")
        self.percent.pack(anchor="e")
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(8, 14))

        output_frame = ttk.Frame(frame)
        output_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(output_frame)
        scrollbar.pack(side="right", fill="y")
        self.output = tk.Text(
            output_frame,
            height=12,
            state="disabled",
            wrap="word",
            yscrollcommand=scrollbar.set,
        )
        self.output.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.output.yview)

        self.close_button = ttk.Button(
            frame, text="Close", command=self._close_window, state="disabled"
        )
        self.close_button.pack(anchor="e", pady=(12, 0))

        threading.Thread(target=self._run_command, daemon=True).start()
        self.root.after(100, self._drain_messages)

    def _run_command(self) -> None:
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.messages.put(("output", line.rstrip()))
            return_code = self.process.wait()
            self.messages.put(("finished", return_code))
        except OSError as exc:
            self.messages.put(("error", str(exc)))

    def _drain_messages(self) -> None:
        try:
            while True:
                message_type, value = self.messages.get_nowait()
                if message_type == "output":
                    self._append_output(str(value))
                    self._update_progress(str(value))
                elif message_type == "finished":
                    self._finish(int(value))
                else:
                    self._append_output(f"Error: {value}")
                    self._finish(1)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_messages)

    def _update_progress(self, text: str) -> None:
        match = re.search(r"PROGRESS\s+(\d+)\s+(\d+)(?:\s+(.*))?", text)
        if match:
            current, total = int(match.group(1)), max(1, int(match.group(2)))
            percent = min(100, max(0, current * 100 / total))
            if self.title_text == "TXT to JSON":
                self.progress["value"] = percent
                self.percent.config(text=f"{percent:.0f}%")
                self.status.config(text=match.group(3) or "Converting TXT files to JSON...")
                return
            if self.title_text in {
                "PDF to TXT to JSON",
                "PDF to TXT to JSON and Train Offline AI",
            }:
                message = match.group(3) or ""
                if "Training" in message or "epoch" in message.lower():
                    self.training_phase = True
                if self.training_phase:
                    percent = 50 + percent / 2
                elif "Converted" in message:
                    percent = 50 + percent / 2
                elif "Extracted" in message:
                    percent /= 2
                else:
                    percent /= 2
            self.progress["value"] = percent
            self.percent.config(text=f"{percent:.0f}%")
            self.status.config(text=match.group(3) or "Working...")
        elif text:
            self.status.config(text=text)

    def _append_output(self, text: str) -> None:
        self.output.config(state="normal")
        self.output.insert("end", text + "\n")
        self.output.see("end")
        self.output.config(state="disabled")

    def _finish(self, return_code: int) -> None:
        if return_code == 0:
            self.progress["value"] = 100
            self.percent.config(text="100%")
            self.status.config(text="Completed successfully.")
            self.root.after(1500, self._close_window)
        else:
            self.status.config(text=f"Failed with exit code {return_code}.")
        self.close_button.config(state="normal")

    def _close_window(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        if self.process is not None:
            return self.process.returncode or 0
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command with a progress GUI.")
    parser.add_argument("--title", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--":
        parser.error("a command is required after --")
    command = args.command[1:]
    try:
        return ProgressWindow(command, args.title).run()
    except tk.TclError:
        print(
            "A graphical display is unavailable; continuing in the terminal.",
            file=sys.stderr,
        )
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
