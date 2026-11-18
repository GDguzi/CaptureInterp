#!/bin/zsh
# Build CaptureInterp.app next to this script. Signs with an Apple Development identity when one exists
# (camera/microphone permission then survives rebuilds), otherwise ad hoc.
set -e
cd "$(dirname "$0")"
APP=CaptureInterp.app
mkdir -p $APP/Contents/MacOS $APP/Contents/Resources
cat > $APP/Contents/Info.plist <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>local.captureinterp.CaptureInterp</string>
<key>CFBundleName</key><string>CaptureInterp</string>
<key>CFBundleExecutable</key><string>CaptureInterp</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.1</string>
<key>CFBundleVersion</key><string>1</string>
<key>LSMinimumSystemVersion</key><string>26.0</string>
<key>NSHighResolutionCapable</key><true/>
<key>NSCameraUsageDescription</key><string>读取采集卡或摄像头的画面用于补帧。Reads the capture card's picture for frame interpolation.</string>
<key>NSMicrophoneUsageDescription</key><string>播放采集卡的声音。Plays the capture card's sound.</string>
</dict></plist>
PLIST
# The package's resource bundles (model weights, Metal library) are found through the absolute .build path baked
# into the binary, so keep .build in place after building.
swift build -c release
cp .build/release/CaptureInterp $APP/Contents/MacOS/CaptureInterp
cp -R .build/release/*.bundle $APP/Contents/Resources/ 2>/dev/null || true
ID=$(security find-identity -v -p codesigning | awk -F'"' '/Apple Development/{print $2; exit}')
codesign --force --sign "${ID:--}" --timestamp=none $APP
echo "built $APP"
