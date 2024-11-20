# py_face

This script transform the face and background to the images in assets/faces and assets/backgrounds

the rotation of time of each image is:
- background: 3min
- face: 5min

![Image of output camera](assets/example.png)

# Use
```bash
sudo apt update
sudo apt install python3 python3-pip

sudo apt install v4l2loopback-dkms ffmpeg
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="FakeCam" exclusive_caps=1  

pip install -r requirements.txt

sudo usermod -aG video $USER
```

