#!/bin/sh

if [ "$CI_WORKFLOW_ID" = "AB4F8343-1ADC-40C0-964A-0DDC07E7A2D5" ]; then
    xcodebuild -downloadComponent metalToolchain -exportPath /tmp/MyMetalExport/
    sed -i '' -e 's/17A5241c/17A5241e/g' /tmp/MyMetalExport/MetalToolchain-17A5241c.exportedBundle/ExportMetadata.plist
    xcodebuild -importComponent metalToolchain -importPath /tmp/MyMetalExport/MetalToolchain-17A5241c.exportedBundle
fi
