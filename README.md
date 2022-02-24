# Rpi

## Cloning

Open a terminal and type the following

```bash
git clone --recursive https://github.com/CZ3004-MDP-Group-27/Rpi
```

## Image Recognition

To start the image recognition server open a terminal and type  the following

```bash
cd image_recognition/yolov5
python image_recognition_server.py --weights trained_models/exp1/best.pt --img 416 --conf 0.5 --source 1
```