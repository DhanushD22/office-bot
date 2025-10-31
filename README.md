# Face Recognition on Jetson Nano using NCNN and OpenCV

This project implements a **real-time face recognition and liveness detection system** on the **NVIDIA Jetson Nano**, leveraging the **NCNN framework** for fast neural network inference and **OpenCV** for image handling and preprocessing.

The application uses:
- RetinaFace for face detection  
- MobileFaceNet for face feature extraction  
- MTCNN for multi-stage face detection  
- Custom liveness detection models

---

## 🧠 Project Overview

This implementation demonstrates:
1. **Face Detection** using NCNN-based RetinaFace/MTCNN.
2. **Face Recognition** using MobileFaceNet embeddings.
3. **Liveness Detection** to identify spoof/fake faces.
4. **Automatic Enrollment** of new faces into the database.
5. **Blur Filtering** to skip unclear images.

The system is lightweight and optimized for **Jetson Nano** but can also be built on **any Linux x86 or ARM64 system** with NCNN and OpenCV installed.

---

You need the following packages on your system:

sudo apt update
sudo apt install -y build-essential git cmake pkg-config

## 📁 Project Structure

Face-Recognition-Jetson-Nano/
├── include/              # Header files (ArcFace, Retina, MTCNN, LiveFace, etc.)
├── models/               # Pre-trained NCNN model files (.param and .bin)
│   ├── mtcnn/
│   ├── retina/
│   ├── mobilefacenet/
│   └── live/
├── src/                  # C++ source files
├── FaceRecognition.cbp   # Code::Blocks project (optional)
└── README.md

# Dependencies

Install all required tools:

sudo apt update
sudo apt install -y build-essential git cmake pkg-config
sudo apt install -y libopencv-dev python3-opencv


# NCNN Installation

The project depends on the NCNN C++ library (from Tencent).

To build NCNN from source:

git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DNCNN_VULKAN=ON ..
make -j4
sudo make install

This installs:
Headers to: /usr/local/include/ncnn/
Static libraries to: /usr/local/lib/

# Building the Project

1. ## Clone this repository:

git clone https://github.com/DhanushD22/office-bot.git
cd Face-Recognition-Jetson-Nano
mkdir build && cd build

then compile everything:

g++ ../src/*.cpp -I../include -I/usr/local/include/ncnn \
    `pkg-config --cflags --libs opencv4` \
    -L/usr/local/lib -lncnn \
    -L~/ncnn/build/install/lib \
    -lglslang -lMachineIndependent -lOSDependent -lGenericCodeGen -lSPIRV -lglslang-default-resource-limits \
    -ldl -lpthread -fopenmp \
    -o FaceRecognition

2. ## Create required directories:

mkdir -p models/mtcnn models/mobilefacenet models/retina models/live img

3. ## Then copy the trained NCNN model files into the models/ directory:
models/
├── mtcnn/
│   ├── det1.param
│   ├── det1.bin
│   ├── det2.param
│   ├── det2.bin
│   ├── det3.param
│   └── det3.bin
├── retina/
│   ├── mnet.25-opt.param
│   └── mnet.25-opt.bin
├── mobilefacenet/
│   ├── mobilefacenet.param
│   └── mobilefacenet.bin
└── live/
    ├── model_1.param
    ├── model_1.bin
    ├── model_2.param
    └── model_2.bin

Tip: You must have the models/ directory inside the same folder where you run the binary (build/).

4. ## Add Sample Images

You can test with any face image(inside build folder). Example:
cd img/
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg
cd ..

5. ## Run the Application

From inside the build/ directory:

./FaceRecognition

If all model paths and folders are correct, you’ll see output like:

OpenCV Version: 4.2.0
Trying to recognize faces
Using Retina
Test living or fake face
Automatic adding strangers to database
Blur filter - only sharp images to database
Found 1 pictures in database.

