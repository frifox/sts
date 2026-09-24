import threading

from sts.persistent_config import PersistentConfig


def test_get_returns_default_when_unset(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    assert pc.get("asr_backend") is None
    assert pc.get("output_volume", "80") == "80"


def test_set_then_get_round_trips(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    pc.set("output_volume", "50")
    assert pc.get("output_volume") == "50"


def test_set_overwrites_an_existing_key(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))
    pc.set("input_device", "1")
    pc.set("input_device", "2")
    assert pc.get("input_device") == "2"


def test_values_survive_reopening_the_same_file(tmp_path):
    db_path = str(tmp_path / "sts.db")
    pc = PersistentConfig(db_path)
    pc.set("voice:xtts", "saved:abc123")
    pc.close()

    reopened = PersistentConfig(db_path)
    assert reopened.get("voice:xtts") == "saved:abc123"


def test_concurrent_writers_do_not_corrupt_the_store(tmp_path):
    pc = PersistentConfig(str(tmp_path / "sts.db"))

    def writer(n):
        for i in range(20):
            pc.set(f"key{n}", str(i))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for n in range(5):
        assert pc.get(f"key{n}") == "19"
