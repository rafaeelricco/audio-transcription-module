import sys
import time
import random
import threading


class ProgressBar:
    """
    A console-based progress bar utility for operation tracking.

    This class provides both deterministic and simulated progress tracking with
    support for multi-step operations and background progress simulation.

    Attributes:
        total_width (int): Visual width of the progress bar in characters
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
            total_width (int): Width of the progress bar in console characters
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
            step (str): Description of the current operation step
            progress (float, optional): Completion percentage (0-100)
        """
        if step != self.current_step:
            if self.current_step and self.progress < 100:
                self.update(self.current_step, 100)
            sys.stdout.write(f"\n==> {step}\n")
            self.current_step = step
            self.progress = 0
            self._step_printed = True

        if progress is not None:
            self.progress = min(100, max(0, progress))
            filled_width = int(self.total_width * self.progress / 100)
            empty_width = self.total_width - filled_width
            bar = "#" * filled_width + "-" * empty_width
            sys.stdout.write(f"\r{bar} {self.progress:.1f}%")
            sys.stdout.flush()

            if self.progress == 100:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._stop_fake_progress = True

    def simulate_progress(self, step, start_from=0, until=80):
        """
        Start simulated progress tracking in the background.

        Args:
            step (str): Description of the operation step
            start_from (float): Initial progress percentage
            until (float): Target progress percentage
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
                time.sleep(random.uniform(0.5, 2))
                if not self._stop_fake_progress:
                    current_progress += random.uniform(0.5, 2)
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
