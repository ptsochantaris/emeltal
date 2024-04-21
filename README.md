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
- [FusionNet 13b (Truthful_DPO_TomGrc)](https://huggingface.co/yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B)
- [Nous Hermes 2 on Mixtral](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO)
- [OpenChat 3.5](https://huggingface.co/openchat/openchat-3.5-0106)

#### Dolphin Series
- [Dolphin 2.7 on Mixtral](https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b)
- [Dolphin 2.8 on TinyLlama](https://huggingface.co/cognitivecomputations/TinyDolphin-2.8-1.1b)
- [Dolphin 2.2 70b](https://huggingface.co/cognitivecomputations/dolphin-2.2-70b)

#### Samantha Series
- [Samantha 7b](https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b)
- [Samantha 70b](https://huggingface.co/cognitivecomputations/Samantha-1.11-70b)

#### Smaug Series
- [Smaug 34b](https://huggingface.co/abacusai/Smaug-34B-v0.1)
- [Smaug 72b](https://huggingface.co/abacusai/Smaug-72B-v0.1)

#### Llama Series
- [Llama 3 70b](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
- [Llama 3 8b](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

#### Coding
- [Dolphin Coder](https://huggingface.co/cognitivecomputations/dolphincoder-starcoder2-15b)
- [Deepseek Coder 33b](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)
- [Deepseek Coder 7b](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
- [Everyone Coder](https://huggingface.co/rombodawg/Everyone-Coder-33b-v2-Base)
- [CodeLlama 70b](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf)

#### Experimental
- [Senku](https://huggingface.co/ShinojiResearch/Senku-70B-Full)
- [MiniCPM-OpenHermes](https://huggingface.co/indischepartij/MiniCPM-3B-OpenHermes-2.5-v2)

#### Creative
- [MythoMax 13b](https://huggingface.co/Gryphe/MythoMax-L2-13b)
- [Neural Story](https://huggingface.co/NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story)

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
