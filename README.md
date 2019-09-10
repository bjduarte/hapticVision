# hapticVision
Video object detection and depth recognition for hapwrap technology.

![Alt Text](https://media.giphy.com/media/UrnKsm0Xj0lMsFYebR/giphy.gif)
-----
![Alt Text](https://media.giphy.com/media/XDFHWBfwplKtwsEYXI/giphy.gif)
-----
Requirements
-----
This program uses two different Machine Learning Models for depth perception. Within the hapticVision main folder create a new folder called "models".  Download both Monodepth2 and ImageAI and place it in the newly created models folder.

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

**Keras**
>pip3 install Keras

Running the program!
-----
Run the main python method. Place the wanted video inside the video folder and change the video_name variable to the name of the video.


Program outputs the predicted depth, distance, height and width, and angle from camera of detected objects. Note, the program is slightly innaccurate for outputs within 10 feet of the Camera.



