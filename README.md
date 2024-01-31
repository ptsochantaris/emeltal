<img src="https://ptsochantaris.github.io/trailer/EmeltalLogo.webp" alt="Logo" width=256 align="right">

Emeltal
====

_The wise cheese_

Local ML voice chat using high-end models, aiming for a self contained, user-friendly out-of-the-box experience as much as possible.

This is a work in progress with frequent updates; [TestFlight builds are available here](https://testflight.apple.com/join/NTIomxyk).

|Selection|Full|Mini|
|---------|----|----|
|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot0.png" alt="Screenshot 0">|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot1.png" alt="Screenshot 1">|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot2.png" alt="Screenshot 2">|

<img src="https://ptsochantaris.github.io/trailer/EmellinkScreenshot.webp" alt="Logo" width=128 align="right">

## Emellink

A light helper app which can run on an iPhone or device with not enough processing power, which automatically detects and connects to Emeltal on the network and provides the same voice interface. [Testflight link for this app is here](https://testflight.apple.com/join/s0EYVO5P)

## Currently supported models

Emeltal offers a hand-picked list of proven open-source high-performance models, aiming to provide the best model for each category/size combination. This list often changes and expands as new models become available, or others are superceeded by much better performing ones. All models run at their maximum context size.

#### General Chat
- [SauerkrautLM-SOLAR](https://huggingface.co/VAGOsolutions/SauerkrautLM-SOLAR-Instruct)
- [Nous Hermes 2 on Mixtral](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)
- [FusionNet 13b (Truthful_DPO_TomGrc)](https://huggingface.co/yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B)
- [OpenChat 3.5](https://huggingface.co/openchat/openchat-3.5-0106)
- [TinyLlama 1.1b](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

#### Dolphin Series
- [Dolphin 2.7 on Mixtral](https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b)
- [Dolphin 2.2 70b](https://huggingface.co/cognitivecomputations/dolphin-2.2-70b)
- [Dolphin 2.6 on Phi2](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2)

#### Coding
- [Deepseek Coder 33b](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)
- [Deepseek Coder 7b](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
- [CodeLlama 70b](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf)

#### Jumbo Size
- [MoMo 72b](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO)

#### Creative
- [MythoMax 13b](https://huggingface.co/Gryphe/MythoMax-L2-13b)

#### Voice Recognition
- [Whisper](https://huggingface.co/ggerganov/whisper.cpp)

## Packages

- Emeltal heavily relies on the [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM processing, and [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for voice recognition.
- Text rendering uses [Ink](https://github.com/JohnSundell/Ink) to convert between Markdown and HTML.
- Uses my [PopTimer](https://github.com/ptsochantaris/pop-timer) for debouncing things.

## License

Released under the terms of the MIT license, see the [LICENSE](LICENSE.txt) file for license rights and limitations (MIT).

All model data which is downloaded locally by the app comes from HuggingFace, and use of the models and data is subject to the respective license of each specific model.

## Copyright

Copyright (c) 2023-2024 Paul Tsochantaris
