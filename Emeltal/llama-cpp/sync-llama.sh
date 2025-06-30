echo ">>> Pulling latest Llama and Whisper repos"

CURRENT=`pwd`
LSRC=~/llama.cpp
WSRC=~/whisper.cpp

cd $LSRC
git pull

cd $WSRC
git pull

cd $CURRENT

echo ">>> Copying files"

cp -r\
 $LSRC/ggml/src/ggml-alloc.c\
 $LSRC/ggml/include/ggml-alloc.h\
 $LSRC/ggml/src/ggml-backend-impl.h\
 $LSRC/ggml/include/ggml-backend.h\
 $LSRC/ggml/src/ggml-backend.cpp\
 $LSRC/ggml/src/ggml-impl.h\
 $LSRC/ggml/src/ggml-backend-reg.cpp\
 $LSRC/ggml/src/ggml-quants.*\
 $LSRC/ggml/src/ggml-common.h\
 $LSRC/ggml/src/ggml.c\
 $LSRC/ggml/src/ggml-threading.*\
 $LSRC/ggml/include/ggml.h\
 $LSRC/ggml/include/ggml-cpp.h\
 $LSRC/ggml/include/ggml-blas.h\
 $LSRC/ggml/include/ggml-cpu.h\
 $LSRC/ggml/include/gguf.h\
 $LSRC/ggml/include/ggml-opt.h\
 $LSRC/ggml/src/ggml-opt.cpp\
 $LSRC/ggml/src/ggml-common.h\
 $LSRC/ggml/src/gguf.cpp\
 $LSRC/ggml/src/ggml-cpu\
 $LSRC/ggml/src/ggml-metal\
 $LSRC/ggml/include/ggml-metal.h\
 $LSRC/ggml/include/ggml-cpu.h\
 $LSRC/ggml/src/ggml-threading.h\
 $LSRC/ggml/src/ggml-blas\
 $LSRC/src/llama-chat.h\
 $LSRC/src/llama-chat.cpp\
 $LSRC/src/llama-io.h\
 $LSRC/src/llama-io.cpp\
 $LSRC/src/llama-memory*\
 $LSRC/src/llama-graph.*\
 $LSRC/src/llama-context.*\
 $LSRC/src/llama-batch.*\
 $LSRC/src/llama-cparams.*\
 $LSRC/src/llama-hparams.*\
 $LSRC/src/llama-adapter.*\
 $LSRC/src/llama-kv-*\
 $LSRC/src/llama-model.*\
 $LSRC/src/llama-model-loader.*\
 $LSRC/src/llama-model-saver.*\
 $LSRC/src/llama-arch.*\
 $LSRC/src/llama-mmap.*\
 $LSRC/src/llama-impl.*\
 $LSRC/src/llama.cpp\
 $LSRC/src/llama-vocab.*\
 $LSRC/src/llama-grammar.*\
 $LSRC/src/llama-sampling.*\
 $LSRC/src/unicode.*\
 $LSRC/src/unicode-data.*\
 $LSRC/include/llama.h\
 $WSRC/src/whisper.cpp\
 $WSRC/src/whisper-arch.h\
 $WSRC/include/whisper.h\
 .

echo ">>> Cleaning up cmake files"

find . -type f \( -name "*.txt" -o -name "*.cmake" \) -exec rm -f {} +

echo ">>> Done"
