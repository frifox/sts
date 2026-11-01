"""The self-signed HTTPS certificate lists its addresses (browsers that trust
it check them), and one made before it did is replaced."""

import shutil
import ssl
import subprocess

import pytest

from sts import cli

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs openssl")


def defaults(monkeypatch, tmp_path):
    cert, key = str(tmp_path / "cert.pem"), str(tmp_path / "key.pem")
    monkeypatch.setattr(cli, "DEFAULT_TLS_CERT", cert)
    monkeypatch.setattr(cli, "DEFAULT_TLS_KEY", key)
    return cert, key


def test_made_with_localhost_and_this_computers_addresses(monkeypatch, tmp_path):
    cert, key = defaults(monkeypatch, tmp_path)
    assert cli.ensure_tls_cert(cert, key)
    names = ssl._ssl._test_decode_cert(cert)["subjectAltName"]
    assert ("DNS", "localhost") in names and ("IP Address", "127.0.0.1") in names
    assert cli.tls_cert_current(cert)


def test_one_without_addresses_is_replaced(monkeypatch, tmp_path):
    cert, key = defaults(monkeypatch, tmp_path)
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                    "-days", "30", "-subj", "/CN=sts"], check=True, capture_output=True)
    assert not cli.tls_cert_current(cert)
    assert cli.ensure_tls_cert(cert, key)
    assert cli.tls_cert_current(cert)


def test_a_named_cert_is_left_alone(monkeypatch, tmp_path):
    defaults(monkeypatch, tmp_path / "default")
    cert, key = str(tmp_path / "mine.pem"), str(tmp_path / "mine.key")
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                    "-days", "30", "-subj", "/CN=mine"], check=True, capture_output=True)
    before = open(cert).read()
    assert cli.ensure_tls_cert(cert, key)
    assert open(cert).read() == before
