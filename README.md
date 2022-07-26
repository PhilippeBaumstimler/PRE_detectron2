# DETECTRON2
## Installation
### Cityscapes Scripts
Install Cytiscapes label script helper
```cmd
python -m pip install cityscapesscripts
```
In the cityscapesscripts module, change the file cityscapesscripts/helpers/labels.py in order to modofy the training Ids of objects into potentially movable objects. Here is the table you should have:
![image](https://user-images.githubusercontent.com/81633901/180443209-ba79a77c-9c58-4480-9474-f84ee2bbe220.png)

### detectron2CustomDataset.py

In the file detectron2CustomDataset.py, at the end of the script, the variable "dir" should have your path to the KITTI segmentation dataset. In our work, we have splitted the KITTI segmentation into 180 training and 20 validation images, although it should work as well with the original dataset.

Now test the script:

```cmd
python detectron2CustomDataset.py
```
As a result, the script will generate and show 3 random images from the KITTI segmentation dataset and a mask of their groundtruth. Three custom datasets are implemented : kitti (the official KITTI segmentation dataset, with 11 labels corresponding to movable objects), kitti8 (the official KITTI segmentation dataset, with the 8 original labels used in pretrained models) and cityscapes_pm (the official Cityscapes dataset, with 11 labels corresponding to movable objects).

The custom dataset cityscapes_pm can be obtained by changing the labels according to our training labels by using cityscapesscripts. Then, the detectron2 builtin_meta.py file need to be changed to fit those labels.

### Detecron2

Install detectron2 repository according to your cuda and pytorch versions:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### Test installation

Test script detectron2Main.py

```cmd
python detectron2Main.py --input /path/to/a/video/sequence/or/a/single/image/from/KITTI --dataset kitti --ckpt /path/to/model.pth
```
Works if no error is printed

## Prediction

Detectron2 can be used to predict semantic instances within an image. The network is pretrained on Cityscapes and fine tuned on the KITTI segmentation dataset. Functions you will need are in the detectron2Main.py file.

The network is initialized in the "init_model_detectron2" function:

```python
#Entry :
#   - dataset: "kitti", "kitti8", ... the name of your custom dataset. If None -> Cityscapes
#   - ckpt: path to the model.pth
#Output :
#   -cfg : the network's configuration parameters
#   -model : the network
def init_model_detectron2(dataset, ckpt=None):
    ...
    return cfg, model
```
Then the prediction is done by using the "get_prediction" function : 

```python
#Entry :
#   - img : Your input image (H,W)
#   - cfg : the configuration of your network
#   - model : the network
#Output :
#   - outputs["instances"] : Output of detectron2 network. This output is a dictionnary
#                  -> outputs["instances"].pred_boxes : Tensor (N,4), bounding boxes for each (N) detected instance, format (x1,y1,x2,y2)
#                  -> outputs["instances"].pred_masks : Tensor (N,H,W), masks for each (N) detected instance in the image (H,W)
#                  -> outputs["instances"].scores : Tensor of N confidence score for each (N) detected instance
#                  -> outputs["instances"].pred_classes  : Tensor of N labels for each (N) detected instance
def get_prediction(img, cfg, model): 
    ...
    return outputs["instances"].to("cpu")
```
The detectron_main.py file shows an exemple of how to make a prediction. For further information, please check the detectron2 API documentation : 
https://detectron2.readthedocs.io/en/latest/index.html

## Training

The training is done in "detectron2Train.py".
    - 

```cmd
python detectron2Train.py --dataset ['kitti', 'cityscapes', 'kitti8'] --ckpt /path/to/checkpoint.pth --output /path/to/output/file --batch_size [default 2] --lr [default 0.001] --max_iter [default 300] --num_classes [default 8] --resume store_true start training from last iteration
```

We use environment variables to adress each dataset, be sure to change their value with the location of your datasets:

```python
if args.dataset == "cityscapes":
    os.environ['CITYSCAPES_DATASET']="/media/nicolas/data/cityscapes_pm/"
    #os.system('echo $CITYSCAPES_DATASET')
    train_mode = "cityscapes_fine_instance_seg_train"
elif args.dataset == "kitti":
    os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
    #os.system('echo $CITYSCAPES_DATASET')
    train_mode = "kitti_seg_instance_train"
    CustomDataset.create_kitti_dataset()
elif args.dataset == "kitti8":
    os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
    #os.system('echo $CITYSCAPES_DATASET')
    train_mode = "kitti_seg_instance_train8"
    CustomDataset.create_kitti_dataset8()
```

This script uses the "detectron2CustomDataset.py" script, both should be in the same file. You can add your own custom dataset in the main function, by following the already written code. The results will be saved in the output file, along with the events file for tensorboard vizualisation.



