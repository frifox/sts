#!/bin/zsh
# Builds Live Translate.app (src-mac/build/Live Translate.app): the app, plus a standalone Python 3.11
# with sts and its MLX voice engine installed, ad-hoc signed. The Python is
# kept in src-mac/build/python between builds, so later builds only reinstall
# sts itself (and anything new in its dependencies).
#
# For a release (.github/workflows/macos-release.yml):
#   STS_VERSION=2026.10.25        the app's version (default 0.1.0)
#   STS_SIGN_IDENTITY="Developer ID Application: ..."
#                                 signed for distribution (sign_app.py) instead of ad hoc
set -euo pipefail
cd "${0:A:h}"
repo="${PWD:h}"

# python-build-standalone: a relocatable CPython (a virtualenv isn't - it
# points at the Python it was made from, by absolute path).
python_release=20260924
python_version=3.11.16
python_archive="cpython-${python_version}+${python_release}-aarch64-apple-darwin-install_only.tar.gz"
python_url="https://github.com/astral-sh/python-build-standalone/releases/download/${python_release}/${python_archive//+/%2B}"

runtime=build/python
if [[ ! -x $runtime/bin/python3 || "$($runtime/bin/python3 -V)" != "Python $python_version" ]]; then
    mkdir -p .cache build
    [[ -f .cache/$python_archive ]] || curl -fL --progress-bar -o ".cache/$python_archive" "$python_url"
    rm -rf $runtime
    tar -xzf ".cache/$python_archive" -C build  # unpacks to build/python
fi
py=$runtime/bin/python3

# setuptools builds in the checkout's build/, and keeps what's there: clear it
# so a module deleted from sts doesn't live on in the app.
rm -rf "$repo/build"
$py -m pip install --disable-pip-version-check -q "${repo}[mlx]"
# A local project with an unchanged version counts as installed: reinstall sts itself every build.
$py -m pip install --disable-pip-version-check -q --no-deps --force-reinstall "$repo"
rm -rf "$repo/build"

version=${STS_VERSION:-0.1.0}
build_number=${${STS_VERSION:+${version//./}}:-1}  # 2026.10.25 -> 20261025
# Python writes nothing into the app at run time (STSProcess sets
# PYTHONDONTWRITEBYTECODE): every module is compiled here, before signing.
$py -m compileall -q -j 0 $runtime/lib >/dev/null || true

# The oldest macOS the app runs on: the newest any installed wheel was built for
# (pip picks mlx's build for the macOS it runs on: 14, 15 or 26).
min_macos=$($py - <<'PY'
import glob, re
tags = [m.groups() for wheel in glob.glob("build/python/lib/python3.11/site-packages/*.dist-info/WHEEL")
        for m in re.finditer(r"macosx_(\d+)_(\d+)", open(wheel).read())]
print("%s.%s" % max(((int(a), int(b)) for a, b in tags), default=(13, 0)))
PY
)

swift build -c release
app="build/Live Translate.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$(swift build -c release --show-bin-path)/STSApp" "$app/Contents/MacOS/LiveTranslate"
swift make_icon.swift build/AppIcon.iconset
iconutil -c icns -o "$app/Contents/Resources/AppIcon.icns" build/AppIcon.iconset
# The Configuration form's rows: the bundled sts's own options.
$py export_options.py > "$app/Contents/Resources/options.json"
ditto $runtime "$app/Contents/Resources/python"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Live Translate</string>
    <key>CFBundleDisplayName</key><string>Live Translate</string>
    <key>CFBundleIdentifier</key><string>com.frifox.sts</string>
    <key>CFBundleExecutable</key><string>LiveTranslate</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundleShortVersionString</key><string>${version}</string>
    <key>CFBundleVersion</key><string>${build_number}</string>
    <key>LSMinimumSystemVersion</key><string>${min_macos}</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <!-- sts runs as a child of the app, so macOS asks for the mic on the app's behalf -->
    <key>NSMicrophoneUsageDescription</key><string>sts listens to the speaker to caption and translate them.</string>
    <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
</dict>
</plist>
PLIST

if [[ -n ${STS_SIGN_IDENTITY:-} ]]; then
    $py sign_app.py "$app" "$STS_SIGN_IDENTITY"
else
    codesign --force --sign - "$app"
fi
# Finder and the Dock cache an app's icon by path: have them pick up this build's.
touch "$app"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$app"
echo "built $PWD/$app ($(du -sh "$app" | cut -f1))"
