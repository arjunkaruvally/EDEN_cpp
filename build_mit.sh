echo "Configuring..."
export CAFFE2_USE_CUDNN=0
cmake -DCMAKE_PREFIX_PATH=/home/gridsan/akaruvally/libtorch -DCAFFE2_USE_CUDNN=0 ..

echo "Building..."
cmake --build . --parallel 8
