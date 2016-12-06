// easy install
sudo apt-get install libopencv-dev python-opencv

// pre-req Installs and build from source

sudo apt-get install build-essential
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
sudo apt-get install liblzma-doc libraw1394-doc tbb-examples libtbb-doc gfortran python-nose python-numpy-dbg python-numpy-doc
sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev libqt4-opengl-dev
sudo apt-get install libgtkglext1 libgtkglext1-dev

sudo apt-get install python-scipy
sudo apt-get install python-matplotlib

// OpenCV
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
git clone https://github.com/opencv/opencv_extra.git

// Building OpenCV from Source Using CMake
cd ~/opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=Release -DWITH_OPENGL=ON  -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j4

sudo make install

// test OpenCV
python
import cv2
cv2.__version__


// Windows install 

Install Python for windows

// get packages form here

http://www.lfd.uci.edu/~gohlke/pythonlibs

// from command line

pip install "basemap-1.0.8-cp35-none-win_amd64.whl"
pip install "opencv_python-3.1.0-cp35-cp35m-win_amd64.whl"
pip install imutils

