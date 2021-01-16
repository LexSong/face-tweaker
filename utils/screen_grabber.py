import numpy as np
import win32gui
from PIL import ImageGrab


class WindowNotFoundError(Exception):
    pass


def _get_all_window_titles():
    def callback(hwnd, results):
        results.append((hwnd, win32gui.GetWindowText(hwnd)))

    results = []
    win32gui.EnumWindows(callback, results)
    return results


def get_hwnd_by_title_prefix(title_prefix):
    for hwnd, title in _get_all_window_titles():
        if title.startswith(title_prefix):
            return hwnd
    raise WindowNotFoundError(f'Can\'t find window title starts with "{title_prefix}"')


class ScreenGrabber:
    def __init__(self, title_prefix):
        self.hwnd = get_hwnd_by_title_prefix(title_prefix)

    def grab_screenshot(self):
        win32gui.SetForegroundWindow(self.hwnd)
        bbox = win32gui.GetWindowRect(self.hwnd)
        return np.array(ImageGrab.grab(bbox))
