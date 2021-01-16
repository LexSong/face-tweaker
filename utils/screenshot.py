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


def get_screenshot(title_prefix):
    hwnd = get_hwnd_by_title_prefix(title_prefix)
    win32gui.SetForegroundWindow(hwnd)
    bbox = win32gui.GetWindowRect(hwnd)
    return ImageGrab.grab(bbox)
