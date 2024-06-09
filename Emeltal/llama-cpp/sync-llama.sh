CURRENT=`pwd`
LSRC=~/llama.cpp
WSRC=~/whisper.cpp

cd $LSRC
git pull

cd $WSRC
git pull

cd $CURRENT

cp\
 $LSRC/ggml-alloc.c\
 $LSRC/ggml-alloc.h\
 $LSRC/ggml-backend-impl.h\
 $LSRC/ggml-backend.c\
 $LSRC/ggml-backend.h\
 $LSRC/ggml-impl.h\
 $LSRC/ggml-metal.h\
 $LSRC/ggml-metal.m\
 $LSRC/ggml-metal.metal\
 $LSRC/ggml-quants.c\
 $LSRC/ggml-quants.h\
 $LSRC/ggml-common.h\
 $LSRC/ggml.c\
 $LSRC/ggml.h\
 $LSRC/llama.cpp\
 $LSRC/llama.h\
 $LSRC/unicode.h\
 $LSRC/unicode.cpp\
 $LSRC/unicode-data.h\
 $LSRC/unicode-data.cpp\
 $LSRC/sgemm.h\
 $LSRC/sgemm.cpp\
 $WSRC/whisper.cpp\
 $WSRC/whisper.h\
 $WSRC/whisper-mel.hpp\
 .

patch ggml-metal.m < ggml-metal.diff
