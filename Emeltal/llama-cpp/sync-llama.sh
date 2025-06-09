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
 $LSRC/ggml/src/ggml-quants.c\
 $LSRC/ggml/src/ggml-quants.h\
 $LSRC/ggml/src/ggml-common.h\
 $LSRC/ggml/src/ggml.c\
 $LSRC/ggml/src/ggml-threading.h\
 $LSRC/ggml/src/ggml-threading.cpp\
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
 $LSRC/src/llama-memory.h\
 $LSRC/src/llama-memory.cpp\
 $LSRC/src/llama-graph.h\
 $LSRC/src/llama-graph.cpp\
 $LSRC/src/llama-context.h\
 $LSRC/src/llama-context.cpp\
 $LSRC/src/llama-batch.h\
 $LSRC/src/llama-batch.cpp\
 $LSRC/src/llama-cparams.h\
 $LSRC/src/llama-cparams.cpp\
 $LSRC/src/llama-hparams.h\
 $LSRC/src/llama-hparams.cpp\
 $LSRC/src/llama-adapter.h\
 $LSRC/src/llama-adapter.cpp\
 $LSRC/src/llama-kv-*.h\
 $LSRC/src/llama-kv-*.cpp\
 $LSRC/src/llama-model.h\
 $LSRC/src/llama-model.cpp\
 $LSRC/src/llama-model-loader.h\
 $LSRC/src/llama-model-loader.cpp\
 $LSRC/src/llama-model-saver.h\
 $LSRC/src/llama-model-saver.cpp\
 $LSRC/src/llama-arch.h\
 $LSRC/src/llama-arch.cpp\
 $LSRC/src/llama-mmap.h\
 $LSRC/src/llama-mmap.cpp\
 $LSRC/src/llama-impl.h\
 $LSRC/src/llama-impl.cpp\
 $LSRC/src/llama.cpp\
 $LSRC/src/llama-vocab.h\
 $LSRC/src/llama-vocab.cpp\
 $LSRC/src/llama-grammar.h\
 $LSRC/src/llama-grammar.cpp\
 $LSRC/src/llama-sampling.h\
 $LSRC/src/llama-sampling.cpp\
 $LSRC/src/unicode.h\
 $LSRC/src/unicode.cpp\
 $LSRC/src/unicode-data.h\
 $LSRC/src/unicode-data.cpp\
 $LSRC/include/llama.h\
 $WSRC/src/whisper.cpp\
 $WSRC/src/whisper-arch.h\
 $WSRC/include/whisper.h\
 .

echo ">>> Cleaning up cmake files"

find . -type f \( -name "*.txt" -o -name "*.cmake" \) -exec rm -f {} +

echo ">>> Done"
