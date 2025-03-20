import sys
import time
import random
import threading
import shutil


class ProgressBar:
    """
    A console-based progress bar utility for operation tracking.

    This class provides both deterministic and simulated progress tracking with
    support for multi-step operations and background progress simulation.

    Attributes:
        total_width (int): fallback width da barra caso o terminal não informe o tamanho real.
        current_step (str): Description of the current operation
        progress (float): Current completion percentage
        _stop_fake_progress (bool): Control flag for simulated progress
        _is_running_fake (bool): Status flag for simulated progress
        _current_thread (Thread): Background thread for progress simulation
        _step_printed (bool): Track if step description has been displayed
    """

    def __init__(self, total_width=80):
        """
        Initialize the progress bar with specified width.
        Args:
            total_width (int): Width fallback da barra, em caracteres, se não for possível
                               detectar a largura do terminal.
        """
        self.total_width = total_width
        self.current_step = ""
        self.progress = 0
        self._stop_fake_progress = False
        self._is_running_fake = False
        self._current_thread = None
        self._step_printed = False

    def update(self, step, progress=None):
        """
        Update the progress bar state and display.
        Args:
            step (str): Description of the current operation step.
            progress (float, optional): Completion percentage (0-100).
        """
        if step != self.current_step:
            if self.current_step and self.progress < 100:
                self.update(self.current_step, 100)
            sys.stdout.write(f"\n==> {step}\n")
            sys.stdout.flush()
            self.current_step = step
            self.progress = 0
            self._step_printed = True

        if progress is not None:
            self.progress = min(100, max(0, progress))

            term_width = shutil.get_terminal_size((self.total_width, 20)).columns

            extra_space = 8
            bar_width = term_width - extra_space
            if bar_width < 10:
                bar_width = 10

            filled_len = int(bar_width * self.progress / 100)
            if filled_len >= bar_width:
                bar = "=" * bar_width
            else:
                bar = "=" * filled_len + ">" + "-" * (bar_width - filled_len - 1)

            sys.stdout.write("\r" + " " * term_width + "\r")
            sys.stdout.write(f"{bar} {self.progress:6.1f}%")
            sys.stdout.flush()

            if self.progress >= 100:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._stop_fake_progress = True

    def simulate_progress(self, step, start_from=0, until=80):
        """
        Start simulated progress tracking in the background.
        Args:
            step (str): Description of the operation step.
            start_from (float): Initial progress percentage.
            until (float): Target progress percentage.
        """
        if self._current_thread and self._current_thread.is_alive():
            self._stop_fake_progress = True
            self._current_thread.join()

        self._stop_fake_progress = False
        self._is_running_fake = True
        self.update(step, start_from)

        def fake_progress_worker():
            current_progress = start_from
            while not self._stop_fake_progress and current_progress < until:
                if current_progress < until * 0.3:
                    sleep_time = random.uniform(0.05, 0.15)
                    increment = random.uniform(1.0, 2.0)
                elif current_progress > until * 0.8:
                    sleep_time = random.uniform(0.3, 0.6)
                    increment = random.uniform(0.2, 0.8)
                else:
                    sleep_time = random.uniform(0.15, 0.3)
                    increment = random.uniform(0.5, 1.0)
                time.sleep(sleep_time)
                if not self._stop_fake_progress:
                    current_progress += increment
                    self.update(step, current_progress)
            self._is_running_fake = False

        self._current_thread = threading.Thread(target=fake_progress_worker)
        self._current_thread.daemon = True
        self._current_thread.start()

    def wait_for_fake_progress(self):
        """Wait for any ongoing simulated progress to complete."""
        if self._current_thread and self._current_thread.is_alive():
            self._current_thread.join()

    def reset(self):
        """Reset progress state for new file processing"""
        self.current_step = ""
        self.progress = 0
        self._stop_fake_progress = False
        self._is_running_fake = False
        self._step_printed = False
        if self._current_thread and self._current_thread.is_alive():
            self._stop_fake_progress = True
            self._current_thread.join()
