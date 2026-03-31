#!/bin/bash
# helpers/build_audio_helper.sh
#
# Compiles the Swift ScreenCaptureKit audio helper.
# Run once before packaging or developing on macOS 13+.
#
# Usage:
#   chmod +x helpers/build_audio_helper.sh
#   ./helpers/build_audio_helper.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE="$SCRIPT_DIR/audio_helper.swift"
OUTPUT="$PROJECT_ROOT/audio_helper"

# Check macOS version
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
if [ "$MACOS_MAJOR" -lt 13 ]; then
  echo "⚠️  macOS 13+ required to build ScreenCaptureKit helper."
  echo "   On macOS $MACOS_MAJOR, SOVA will use BlackHole instead."
  exit 0
fi

echo "Building audio_helper..."

swiftc "$SOURCE" \
  -o "$OUTPUT" \
  -framework ScreenCaptureKit \
  -framework CoreAudio \
  -framework AVFoundation \
  -target arm64-apple-macos13.0

# Also build for Intel if running on Apple Silicon (universal binary)
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  echo "Building Intel slice for universal binary..."
  swiftc "$SOURCE" \
    -o "${OUTPUT}_x86" \
    -framework ScreenCaptureKit \
    -framework CoreAudio \
    -framework AVFoundation \
    -target x86_64-apple-macos13.0

  lipo -create -output "$OUTPUT" "$OUTPUT" "${OUTPUT}_x86"
  rm "${OUTPUT}_x86"
  echo "Universal binary created."
fi

chmod +x "$OUTPUT"
echo "Done: $OUTPUT"