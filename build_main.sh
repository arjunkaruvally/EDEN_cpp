echo "Configuring..."
export CAFFE2_USE_CUDNN=1
cmake -DCMAKE_PREFIX_PATH=/media/arjun/SSD/libtorch -DCAFFE2_USE_CUDNN=1 ..

echo "Building..."
cmake --build . --parallel 8
