"""The caption page listens on all interfaces, unless asked for this computer only."""

from sts.cli import build_parser, web_host


def _host(*argv):
    return web_host(build_parser().parse_args(list(argv)))


def test_all_interfaces_by_default():
    assert _host() == "0.0.0.0"


def test_this_computer_only_on_request():
    assert _host("--local-only") == "127.0.0.1"


def test_an_explicit_address_wins():
    assert _host("--web-host", "192.168.1.5") == "192.168.1.5"
    assert _host("--web-host", "::", "--local-only") == "::"


def test_only_this_computer_gets_in_when_listening_on_all_interfaces_for_it():
    from sts.web.server import is_loopback

    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("::ffff:127.0.0.1")
    assert not is_loopback("192.168.1.20") and not is_loopback("10.0.0.5") and not is_loopback(None)


def test_a_privileged_port_on_macos_falls_back_to_all_interfaces_kept_local(monkeypatch):
    from sts import cli

    def unavailable(host, port):  # macOS, unprivileged: port 80 on 127.0.0.1 alone is refused
        return "Permission denied" if host == "127.0.0.1" and port < 1024 else None

    monkeypatch.setattr(cli, "port_unavailable", unavailable)
    assert cli.local_bind([80, 443]) == ("0.0.0.0", True)
    assert cli.local_bind([8765]) == ("127.0.0.1", False)
