from sts.cli import saved_device
from sts.persistent_config import PersistentConfig


def _config(tmp_path, **values):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    for key, value in values.items():
        pc.set(key, value)
    return pc


def _rejects(device):
    raise RuntimeError(f"Error querying device {device}")


def test_a_usable_saved_device_is_restored(tmp_path):
    pc = _config(tmp_path, input_device="3")
    checked = []

    assert saved_device(pc, "input_device", checked.append) == 3
    assert checked == [3]  # it was actually tried, not just trusted


def test_a_saved_device_that_no_longer_exists_falls_back_to_the_system_default(tmp_path, capsys):
    """Real failure from live use: input device 13 was saved, then the
    device was gone (unplugged, or indices shifted), and `sts` crashed with
    a PortAudioError from pipeline.start() - after every model had loaded."""
    pc = _config(tmp_path, input_device="13")

    assert saved_device(pc, "input_device", _rejects) is None
    assert "input device '13' can't be used" in capsys.readouterr().out


def test_a_stale_saved_device_is_kept_so_it_returns_when_the_device_does(tmp_path):
    pc = _config(tmp_path, input_device="13")

    saved_device(pc, "input_device", _rejects)

    assert pc.get("input_device") == "13"


def test_nothing_saved_or_saved_as_system_default_means_none(tmp_path):
    pc = _config(tmp_path, output_device="")  # "" is how the system default is stored

    assert saved_device(pc, "input_device", _rejects) is None   # never saved
    assert saved_device(pc, "output_device", _rejects) is None  # saved as default; not even checked
    assert saved_device(None, "input_device", _rejects) is None  # persistence disabled


def test_a_garbled_saved_value_falls_back_too(tmp_path):
    pc = _config(tmp_path, input_device="not-a-number")

    assert saved_device(pc, "input_device", lambda d: None) is None
