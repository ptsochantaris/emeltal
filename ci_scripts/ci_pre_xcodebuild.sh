#!/bin/sh

if [ "$CI_WORKFLOW_ID" = "AB4F8343-1ADC-40C0-964A-0DDC07E7A2D5" ]; then
    xcodebuild -downloadComponent metalToolchain
fi
