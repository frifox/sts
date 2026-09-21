import sounddevice as sd

from sts.audio import devices


def fake_devices():
    return [
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2, "hostapi": 0},
        {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0, "hostapi": 0},
        {"name": "Headset", "max_input_channels": 1, "max_output_channels": 2, "hostapi": 0},
    ]


def test_devices_are_split_into_inputs_and_outputs(monkeypatch):
    monkeypatch.setattr(sd, "query_devices", lambda kind=None: fake_devices() if kind is None else {"index": 1})
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "CoreAudio"}])
    result = devices.describe_devices(input_device=2, output_device=None)
    assert [d["index"] for d in result["inputs"]] == [1, 2]      # only devices that can capture
    assert [d["index"] for d in result["outputs"]] == [0, 2]     # only devices that can play
    assert result["inputs"][0]["name"] == "Microphone"           # single host API: plain names
    assert (result["input"], result["output"]) == (2, None)      # None means "system default"
    assert result["default_input"] == 1


def test_host_api_shown_only_when_there_are_several(monkeypatch):
    monkeypatch.setattr(sd, "query_devices", lambda kind=None: fake_devices() if kind is None else {"index": 0})
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "MME"}, {"name": "WASAPI"}])
    result = devices.describe_devices(None, None)
    assert result["outputs"][0]["name"] == "Speakers (MME)"      # Windows lists each card once per host API


def test_a_failing_default_lookup_does_not_break_the_list(monkeypatch):
    def query(kind=None):
        if kind is None:
            return fake_devices()
        raise sd.PortAudioError("no default device")

    monkeypatch.setattr(sd, "query_devices", query)
    monkeypatch.setattr(sd, "query_hostapis", lambda: [{"name": "ALSA"}])
    result = devices.describe_devices(None, None)
    assert result["default_input"] is None and len(result["inputs"]) == 2
