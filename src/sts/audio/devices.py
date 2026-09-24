"""Lists the machine's audio devices for the configuration dialog, and checks a
device will open with the settings the pipeline needs before it is switched to."""

import sounddevice as sd


def describe_devices(input_device: int | None, output_device: int | None) -> dict:
    """Input and output devices (index + display name), the current selection
    (None = the system default), and which devices the system defaults are."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    show_api = len(hostapis) > 1  # on Windows the same card appears once per host API

    def label(device: dict) -> str:
        return f"{device['name']} ({hostapis[device['hostapi']]['name']})" if show_api else device["name"]

    def default_index(kind: str) -> int | None:
        try:
            return int(sd.query_devices(kind=kind)["index"])
        except Exception:
            return None

    return {
        "type": "devices",
        "inputs": [{"index": i, "name": label(d)} for i, d in enumerate(devices) if d["max_input_channels"] > 0],
        "outputs": [{"index": i, "name": label(d)} for i, d in enumerate(devices) if d["max_output_channels"] > 0],
        "input": input_device,
        "output": output_device,
        "default_input": default_index("input"),
        "default_output": default_index("output"),
    }


def check_input(device: int | None, sample_rate: int, channels: int) -> None:
    """Raises if the device can't capture at all - not necessarily at
    `sample_rate` exactly. PortAudio's raw ALSA "hw" access (no "plug"
    layer) means many real sound cards - an onboard Line-In codec, say -
    only accept a couple of fixed hardware rates like 44100/48000 and
    flatly reject an arbitrary one like the pipeline's 16kHz; AudioCapture
    falls back to the device's own rate and resamples, so this check
    should accept exactly what AudioCapture will actually end up doing."""
    try:
        sd.check_input_settings(device=device, channels=channels, samplerate=sample_rate, dtype="float32")
    except Exception:
        info = sd.query_devices(device, "input")
        sd.check_input_settings(device=device, channels=channels, samplerate=info["default_samplerate"], dtype="float32")


def check_output(device: int | None, sample_rate: int) -> None:
    """Raises if the device can't play mono audio at this rate."""
    sd.check_output_settings(device=device, channels=1, samplerate=sample_rate, dtype="float32")


def reinitialize_audio_system() -> None:
    """Rebuilds PortAudio's view of the machine's audio devices.

    PortAudio enumerates devices once, when sounddevice is imported, and
    keeps using that snapshot. After the machine sleeps and wakes, or a
    device is plugged in/removed or the default output changes, the snapshot
    is stale: streams opened against it fail with errors like
    "Internal PortAudio error -9986", and an open input stream may just stop
    delivering. Terminating and re-initializing PortAudio re-enumerates
    everything (the same thing a restart of the process would do).

    Every stream must be closed *before* this is called - terminating
    invalidates them, and touching one afterwards can crash. Uses
    sounddevice's private _initialize/_terminate, which is the standard way
    to do this (there's no public API for it).
    """
    sd._terminate()
    sd._initialize()
