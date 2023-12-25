<img src="https://ptsochantaris.github.io/trailer/EmeltalLogo.webp" alt="Logo" width=256 align="right">

Emeltal
====

Local ML voice chat using high-end models, aiming for a self contained, user-friendly out-of-the-box experience as much as possible.

This is a work in progress with frequent updates; [TestFlight builds are available here](https://testflight.apple.com/join/NTIomxyk).

|Selection|Full|Mini|
|---------|----|----|
|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot0.png" alt="Screenshot 0">|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot1.png" alt="Screenshot 1">|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot2.png" alt="Screenshot 2">|

## Emellink

A light helper app which can run on an iPhone or device with not enough processing power, which automatically detects and connects to Emeltal on the network and provides the same voice interface. [Testflight link for this app is here](https://testflight.apple.com/join/s0EYVO5P)

## Packages
- Emeltal heavily relies on the [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) libraries for ML processing and voice recognition.
- Text rendering is via the [Swift Markdown UI](https://github.com/gonzalezreal/swift-markdown-ui) package.

## License

Released under the terms of the MIT license, see the [LICENSE](LICENSE.txt) file for license rights and limitations (MIT).

All model data which is downloaded locally by the app comes from HuggingFace, and use of the models and data is subject to the respective license of each specific model.

## Copyright

Copyright (c) 2023 Paul Tsochantaris
