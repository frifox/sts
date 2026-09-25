"""macOS microphone permission (TCC) for local input devices.

Without it, CoreAudio doesn't fail: the input stream opens and delivers
silence, so sts looked fine but never heard anything. The permission
belongs to the app sts was started from (Terminal, iTerm, an IDE), not to
Python itself. macOS only shows its prompt while the answer is still "not
determined" - opening the input stream does that on its own - and never
asks again once denied, so all sts can do then is point at the setting.

Talks to AVFoundation through the Objective-C runtime via ctypes, so it
needs no pyobjc in either virtualenv.
"""

import ctypes
import ctypes.util
import subprocess
import sys

# AVAuthorizationStatus values.
_STATUSES = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}

SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"

DENIED_MESSAGE = (
    "macOS is blocking microphone access for the app sts was started from, so every "
    "input device records silence. Allow it in System Settings > Privacy & Security > "
    "Microphone, then restart sts."
)


def mic_permission() -> str | None:
    """"authorized", "denied", "restricted" or "not_determined"; None when
    not on macOS or the status can't be read."""
    if sys.platform != "darwin":
        return None
    try:
        av = ctypes.CDLL("/System/Library/Frameworks/AVFoundation.framework/AVFoundation")
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ("objc_msgSend", objc)
        )
        status = send(
            objc.objc_getClass(b"AVCaptureDevice"),
            objc.sel_registerName(b"authorizationStatusForMediaType:"),
            ctypes.c_void_p.in_dll(av, "AVMediaTypeAudio"),
        )
    except Exception:
        return None
    return _STATUSES.get(status)


def is_blocked(status: str | None) -> bool:
    return status in ("denied", "restricted")


def open_mic_settings() -> None:
    """Opens System Settings at Privacy & Security > Microphone."""
    try:
        subprocess.Popen(["open", SETTINGS_URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
