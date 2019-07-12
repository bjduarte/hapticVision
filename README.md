# hapticVision
Video object detection and depth recognition for hapwrap technology.


Requirements
-----
This program uses two different Machine Learning Models for depth perception. Download each and place it in the models folder.

[Click Here To Download the Models](https://drive.google.com/drive/folders/1xttyp-wezKU9RcIfaCJewzUEHkljIsBt?usp=sharing)

1.Monodepth2 - Used for depth perception
https://github.com/nianticlabs/monodepth2

2.ImageAI - Used for object detection
https://github.com/OlafenwaMoses/ImageAI

Dependencies
-----
Python 3.5.1 or later

**TensorFlow 1.4 or later**
> pip3 install tensorflow

**OpenCV**
>pip3 install opencv-python

>pip3 install cv2

**numpy**
>pip3 install numpy

**matplotlib**
>pip3 install matplotlib

**PIL**
>pip3 install PIL

**ImageAI**
>pip3 install ImageAi

**Pytorch**
>https://pytorch.org/

In order to run this on a video place the given video into the assets\videos folder and change video_converter.getPicturesFromVideo('test.mp4') to the name of the video.

Program outputs the predicted depth, distance, height and width, and angle from camera of detected objects. Note, the program has problems detecing these outputs for objects within 10 feet of the Camera.


Place in Models Folder
https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zi
