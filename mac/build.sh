#!/bin/zsh
# Builds STS.app (mac/build/STS.app): the release binary wrapped in an app
# bundle, ad-hoc signed.
set -euo pipefail
cd "${0:A:h}"
repo="${PWD:h}"

swift build -c release
app=build/STS.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
cp "$(swift build -c release --show-bin-path)/STSApp" "$app/Contents/MacOS/STS"

cat > "$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>STS</string>
    <key>CFBundleDisplayName</key><string>STS</string>
    <key>CFBundleIdentifier</key><string>com.frifox.sts</string>
    <key>CFBundleExecutable</key><string>STS</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>0.1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSPrincipalClass</key><string>NSApplication</string>
    <!-- sts runs as a child of the app, so macOS asks for the mic on the app's behalf -->
    <key>NSMicrophoneUsageDescription</key><string>sts listens to the speaker to caption and translate them.</string>
    <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
    <!-- The default working directory: the checkout this was built from -->
    <key>STSRepoDirectory</key><string>$repo</string>
</dict>
</plist>
PLIST

codesign --force --sign - "$app"
echo "built $PWD/$app"

