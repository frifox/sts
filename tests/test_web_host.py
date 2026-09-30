"""The caption page listens on this computer only, unless asked otherwise."""

from sts.cli import build_parser, web_host


def _host(*argv):
    return web_host(build_parser().parse_args(list(argv)))


def test_this_computer_only_by_default():
    assert _host() == "127.0.0.1"


def test_all_interfaces_on_request():
    assert _host("--all-interfaces") == "0.0.0.0"


def test_an_explicit_address_wins():
    assert _host("--web-host", "192.168.1.5") == "192.168.1.5"
    assert _host("--web-host", "::", "--all-interfaces") == "::"
