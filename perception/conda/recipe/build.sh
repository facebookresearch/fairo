#!/bin/sh
which python
python -V
which $CONDA_PYTHON_EXE
$CONDA_PYTHON_EXE -V

mkdir build && cd build

# for libcurl 
export C_INCLUDE_PATH=$CONDA_PREFIX/include
cmake -GNinja \
      -DCMAKE_INSTALL_PREFIX=$PREFIX \
      -DCMAKE_PREFIX_PATH=$PREFIX \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DBUILD_SHARED_LIBS=ON \
      -DENABLE_CCACHE=OFF \
      -DBUILD_WITH_OPENMP=OFF \
      -DFORCE_RSUSB_BACKEND=ON \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      -DBUILD_PYTHON_BINDINGS=ON \
      -DBUILD_TOOLS=ON \
      $SRC_DIR

cmake --build . --config Release
cmake --install . --config Release
#cmake --build . --config Release -- pyrealsense2 pybackend2
## cmake --install . --config Release -- pyrealsense2 pybackend2

cp -v $SRC_DIR/build/wrappers/python/py*.cpython-*.so $SRC_DIR/wrappers/python/pyrealsense2
python $SRC_DIR/wrappers/python/find_librs_version.py $SRC_DIR $SRC_DIR/wrappers/python/pyrealsense2/
# $CONDA_PYTHON_EXE -m pip install -v $SRC_DIR/wrappers/python
python -m pip install -v --prefix=$PREFIX $SRC_DIR/wrappers/python
#cd $SRC_DIR/wrappers/python
#$CONDA_PYTHON_EXE setup.py install --prefix=$PREFIX

