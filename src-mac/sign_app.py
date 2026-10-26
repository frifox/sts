"""Signs Live Translate.app for distribution (build.sh, with STS_SIGN_IDENTITY):
every binary inside the bundled Python, then the app, with the hardened
runtime and a secure timestamp - what notarization asks for. Executables (the
app, python3.11, ...) get entitlements.plist; libraries don't need it.

    python sign_app.py "build/Live Translate.app" "Developer ID Application: ..."

An identity of "-" signs ad hoc (no timestamp): for trying the hardened
runtime locally.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENTITLEMENTS = Path(__file__).with_name("entitlements.plist")
MACHO_MAGIC = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def binaries(root: Path):
    """(path, is_executable) for every Mach-O file under root."""
    for folder, _, files in os.walk(root):
        for name in files:
            path = Path(folder) / name
            if path.is_symlink():
                continue
            with open(path, "rb") as f:
                if f.read(4) not in MACHO_MAGIC:
                    continue
            executable = os.access(path, os.X_OK) and not name.endswith((".so", ".dylib"))
            yield path, executable


def sign(path: Path, identity: str, entitlements: bool) -> None:
    command = ["codesign", "--force", "--sign", identity, "--options", "runtime"]
    if identity != "-":
        command.append("--timestamp")
    if entitlements:
        command += ["--entitlements", str(ENTITLEMENTS)]
    result = subprocess.run(command + [str(path)], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"codesign failed for {path}:\n{result.stderr}")


def main() -> None:
    app, identity = Path(sys.argv[1]), sys.argv[2]
    found = list(binaries(app / "Contents" / "Resources"))
    print(f"signing {len(found)} binaries in {app.name}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(sign, path, identity, executable) for path, executable in found]:
            future.result()
    sign(app, identity, entitlements=True)  # last: seals everything inside
    subprocess.run(["codesign", "--verify", "--strict", "--deep", str(app)], check=True)


if __name__ == "__main__":
    main()
