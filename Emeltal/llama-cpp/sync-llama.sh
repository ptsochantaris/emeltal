CURRENT=`pwd`
LSRC=~/llama.cpp
WSRC=~/whisper.cpp

cd $LSRC
git pull

cd $WSRC
git pull

cd $CURRENT

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
 $LSRC/ggml/src/ggml-aarch64.c\
 $LSRC/ggml/src/ggml-aarch64.h\
 $LSRC/ggml/src/ggml.c\
 $LSRC/ggml/src/ggml-threading.h\
 $LSRC/ggml/src/ggml-threading.cpp\
 $LSRC/ggml/include/ggml.h\
 $LSRC/ggml/include/ggml-cpp.h\
 $LSRC/ggml/include/ggml-blas.h\
 $LSRC/ggml/include/ggml-cpu.h\
 $LSRC/ggml/src/ggml-cpu\
 $LSRC/ggml/src/ggml-metal\
 $LSRC/ggml/include/ggml-metal.h\
 $LSRC/ggml/include/ggml-cpu.h\
 $LSRC/ggml/src/ggml-threading.h\
 $LSRC/ggml/src/ggml-blas\
 $LSRC/src/llama-impl.h\
 $LSRC/src/llama.cpp\
 $LSRC/include/llama.h\
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
 $WSRC/src/whisper.cpp\
 $WSRC/include/whisper.h\
 .
