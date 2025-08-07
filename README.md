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
- [Qwen QwQ] (https://huggingface.co/bartowski/Qwen_QwQ-32B-GGUF)
- [Qwen 2.5 72b] (https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF)
- [Qwen 2.5 32b] (https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF)
- [Qwen 2.5 14b] (https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF)
- [Qwen 2.5 7b] (https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF)

#### Dolphin Series
- [Dolphin 3 on Mistral 24b & R1 dataset](https://huggingface.co/bartowski/cognitivecomputations_Dolphin3.0-R1-Mistral-24B-GGUF)
- [Dolphin 3 on Llama 3.1 8b](https://huggingface.co/cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF)
- [Dolphin 3 on Qwen 2.5 3b](https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-3b-GGUF)
- [Dolphin 3 on Qwen 2.5 1.5b](https://huggingface.co/bartowski/Dolphin3.0-Qwen2.5-1.5B-GGUF)
- [Dolphin 2.9.2 on Qwen 2.5 72b](https://huggingface.co/mradermacher/dolphin-2.9.2-qwen2-72b-i1-GGUF)
- [Dolphin 2.9.3 on Mistral Nemo 12b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf)
- [Dolphin on StarCoder2 15b](https://huggingface.co/cognitivecomputations/dolphincoder-starcoder2-15b)

#### Samantha Series
- [Samantha 1.11 70b](https://huggingface.co/cognitivecomputations/Samantha-1.11-70b)
- [Samantha 1.1 7b](https://huggingface.co/cognitivecomputations/samantha-1.1-westlake-7b)

#### Gemma 3 Series
- [Gemma 3 27b](https://huggingface.co/ggml-org/gemma-3-27b-it-GGUF)
- [Gemma 3 12b](https://huggingface.co/ggml-org/gemma-3-12b-it-GGUF)
- [Gemma 3 4b](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)
- [Gemma 3 1b](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF)

#### Llama Series
- [Llama 4 Scout](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF)
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
- [OpenAI Open weights model](https://huggingface.co/openai/gpt-oss-20b)
- [THUD GLM 4](https://huggingface.co/THUDM/GLM-4-32B-0414)
- [THUD GLM Z1](https://huggingface.co/THUDM/GLM-Z1-32B-0414)
- [Nvidia Llama Nemo](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1)
- [DeepSeek R1 Distill on Llama](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5)
- [Supernova Medius](https://huggingface.co/arcee-ai/SuperNova-Medius)
- [Shuttle 3](https://huggingface.co/shuttleai/shuttle-3)
- [SmolLM 2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Athene V2](https://huggingface.co/Nexusflow/Athene-V2-Chat)
- [Calme](https://huggingface.co/MaziyarPanahi/calme-2.4-rys-78b)

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
