# PRE_detectron2
## Cityscapes Scripts
Install Cytiscapes label script helper
```cmd
python -m pip install cityscapesscripts
```
In the cityscapesscripts module, change the file cityscapesscripts/helpers/labels.py in order to modofy the training Ids of objects into potentially movable objects. Here is the table you should have:
![image](https://user-images.githubusercontent.com/81633901/180443209-ba79a77c-9c58-4480-9474-f84ee2bbe220.png)

## detectron2CustomDataset.py

In the file detectron2CustomDataset.py, at the end of the script, the variable "dir" should have your path to the KITTI segmentation dataset. In our work, we have splitted the KITTI segmentation into 180 training and 20 validation images, although it should work as well with the original dataset.

Now test the script:

```cmd
python detectron2CustomDataset.py
```
As a result, the script will generate and show 3 random images from the dataset and a mask of their groundtruth.

##Detecron2

Install detectron2 repository
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

##Test installation

Test script detectron2

```cmd
python detectron2_main.py --input /path/to/a/video/sequence/or/a/single/image/from/KITTI --dataset kitti --ckpt /path/to/model.pth
```

