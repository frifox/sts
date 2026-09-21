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
    """Raises if the device can't capture with these settings."""
    sd.check_input_settings(device=device, channels=channels, samplerate=sample_rate, dtype="float32")


def check_output(device: int | None, sample_rate: int) -> None:
    """Raises if the device can't play mono audio at this rate."""
    sd.check_output_settings(device=device, channels=1, samplerate=sample_rate, dtype="float32")
