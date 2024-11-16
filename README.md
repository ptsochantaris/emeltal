<img src="https://ptsochantaris.github.io/trailer/EmeltalLogo.webp" alt="Logo" width=256 align="right">

Emeltal
====

_The wise cheese_

Local ML voice chat using high-end models, aiming for a self contained, user-friendly out-of-the-box experience as much as possible.

This is a work in progress with frequent updates; [TestFlight builds are available here](https://testflight.apple.com/join/NTIomxyk) for macOS, iOS and visionOS.

|Selection|Full|Mini|
|---------|----|----|
|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot0.png" alt="Screenshot 0">|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot1.png" alt="Screenshot 1">|<img src="https://ptsochantaris.github.io/trailer/EmeltalScreenshot2.png" alt="Screenshot 2">|

<img src="https://ptsochantaris.github.io/trailer/EmellinkScreenshot.webp" alt="Logo" width=128 align="right">

## Emellink

A light helper app which can run on an iPhone or device with not enough processing power, which automatically detects and connects to Emeltal on the network and provides the same voice interface. [Testflight link for this app is here](https://testflight.apple.com/join/s0EYVO5P)

## Currently supported models

Emeltal offers a curated list of proven open-source high-performance models, aiming to provide the best model for each category/size combination. This list often changes as new models become available, or others are superceeded by much better performing ones. Most models (with the exception of certain extremely large variants, which are capped at 16384 tokens) run at their maximum context size.

#### Qwen Series
- [Qwen 2.5 72b] (https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF)
- [Qwen 2.5 32b] (https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF)
- [Qwen 2.5 14b] (https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF)
- [Qwen 2.5 7b] (https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF)

#### Dolphin Series
- [Dolphin 2.9.2 on Qwen 2.5](https://huggingface.co/mradermacher/dolphin-2.9.2-qwen2-72b-i1-GGUF)
- [Dolphin 2.7 on Mixtral](https://huggingface.co/cognitivecomputations/dolphin-2.7-mixtral-8x7b)
- [Dolphin 2.9.3 on Mistral Nemo](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf)
- [Dolphin 2.8.1 on TinyLlama](https://huggingface.co/cognitivecomputations/TinyDolphin-2.8-1.1b)

#### Samantha Series
- [Samantha 1.11 70b](https://huggingface.co/cognitivecomputations/Samantha-1.11-70b)
- [Samantha 1.1 7b](https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b)

#### Llama Series
- [Llama 3.1 70b](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct)
- [Llama 3.1 8b](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- [Llama 3.2 3b](https://huggingface.co/meta-llama/Meta-Llama-3.2-8B-Instruct)
- [Llama 3.2 1b](https://huggingface.co/meta-llama/Meta-Llama-3.2-8B-Instruct)

#### Coding
- [Dolphin Coder](https://huggingface.co/cognitivecomputations/dolphincoder-starcoder2-15b)
- [Deepseek Coder 33b](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct)
- [Deepseek Coder 7b](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
- [Everyone Coder](https://huggingface.co/rombodawg/Everyone-Coder-33b-v2-Base)
- [CodeLlama 70b](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf)
- [Codestral](https://huggingface.co/mistralai/Codestral-22B-v0.1)
- [Qwen 2.5 Coder](https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF)

#### Creative
- [MythoMax 13b](https://huggingface.co/Gryphe/MythoMax-L2-13b)
- [Neural Story](https://huggingface.co/NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story)

#### Other & Experimental
- [Supernova Medius](https://huggingface.co/arcee-ai/SuperNova-Medius)
- [Shuttle 3](https://huggingface.co/shuttleai/shuttle-3)
- [SmolLM 2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Athene V2](https://huggingface.co/Nexusflow/Athene-V2-Chat)

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
