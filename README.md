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

The custom dataset cityscapes_pm can be obtained by changing the labels according to our training labels by using cityscapesscripts. It can be achieved by using "createCityscapesTrainDataset.py". Make sure to declare the environmental variable 'CITYSCAPES_DATASET' with the path to the original cityscapes dataset. This script will create a new cityscapes dataset with your training labels. Then, the detectron2 builtin_meta.py file need to be changed to fit those labels.

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

Another example of how to do prediction can be found in "detectron2Predict.py". The script command is:

```cmd
python detectron2Predict.py --dataset ['kitti', 'cityscapes', 'kitti8'] --ckpt /path/to/checkpoint.pth --output /path/to/output/file --input /path/to/a/sequence/or/a/single/file --num_classes [default 8]

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

## Evaluation

The evaluation is done in "detectron2Evluate.py"

```cmd
python detectron2Train.py --dataset ['kitti', 'cityscapes', 'kitti8'] --ckpt /path/to/checkpoint.pth --output /path/to/output/file --batch_size 1 --num_classes [default 8]
```

Prints instance segmentation metrics (mAP) on the evaluation loader of your dataset. 

# Tracking : SORT (Simple Online and Realtime Tracking algorithm)
## Detectron2 x SORT

An example of how to use the the SORT algorithm with detectron2 can be found in "detectron2ObjectTracking.py"

```cmd
python detectron2ObjectTracking.py --input /path/to/a/sequence/file --dataset ["kitti,"cityscapes",kitti8"] --ckpt /path/to/detectron/model.pth --num_classes [kitti and cityscape: 11 / kitti8: 8] --output /path/to/output/file --max_age 1 [--predict_on] 
```
Options:
    --max_age: int>0, maximal age of each tracklets
    --predict_on: allow SORT to predict when the detection is lost

We modified the sort.py file in order to get the predicted masks of detectron2 instead of the one predicted by SORT. You can check the original implementation here: https://github.com/abewley/sort

# Depth/Flow Statistics on KITTI_raw

We preprocessed the KITTI_raw in order to have statistical data to use for motion detection. The file structure should be like this:

```
/KITTI_raw/
    | -- depth_distribution/
    |       | -- quantile_depth_distribution.npy        #size [10,H,W]
    |       | -- linear_depth_ditribution.npy        #size [10,H,W]
    |       | -- pixel_mean_depth.npy        #size [H,W]
    | -- rigid_flow_distribution/
    |       | -- global_mean_flow.npy        #size [H,W]
    |       | -- global_var_flow.npy         #size [H,W]
    |       | -- quantile_mean_flow.npy      #size [10,H,W]
    |       | -- quantile_var_flow.npy       #size [10,H,W]
    |       | -- linear_mean_flow.npy        #size [10,H,W]
    |       | -- linear_var_flow.npy         #size [10,H,W]
    |       | -- tot_frame.npy      #size [1]
    | -- rigid_flow_sequence/
    |       | -- global_mean_flow.npy        #size [H,W]
    |       | -- global_var_flow.npy         #size [H,W]
    |       | -- quantile_mean_flow.npy      #size [10,H,W]
    |       | -- quantile_var_flow.npy       #size [10,H,W]
    |       | -- linear_mean_flow.npy        #size [10,H,W]
    |       | -- linear_var_flow.npy         #size [10,H,W]
    |       | -- tot_frame.npy      #size [1]
    | -- rigid_flow_distribution2/
    |       | -- global_mean_flow.npy        #size [H,W]
    |       | -- global_var_flow.npy         #size [H,W]
    |       | -- quantile_mean_flow.npy      #size [10,H,W]
    |       | -- quantile_var_flow.npy       #size [10,H,W]
    |       | -- linear_mean_flow.npy        #size [10,H,W]
    |       | -- linear_var_flow.npy         #size [10,H,W]
    |       | -- quantile_flow_count.npy     #size [10,H,W]
    |       | -- linear_flow_count.npy       #size [10,H,W] 
    |       | -- tot_frame.npy       #size [1]
    | -- rigid_flow_sequence2/
    |       | -- global_mean_flow.npy        #size [H,W]
    |       | -- global_var_flow.npy         #size [H,W]
    |       | -- quantile_mean_flow.npy      #size [10,H,W]
    |       | -- quantile_var_flow.npy       #size [10,H,W]
    |       | -- linear_mean_flow.npy        #size [10,H,W]
    |       | -- linear_var_flow.npy         #size [10,H,W]
    |       | -- quantile_flow_count.npy     #size [10,H,W]
    |       | -- linear_flow_count.npy       #size [10,H,W] 
    |       | -- tot_frame.npy       #size [1]
```

## Depth distribution
The depth distribution has been obtained with the script "depthDataProcess.py". The data are saved in a numpy array file, in a linear or a quantile format.

## Rigid Flow statistics
The rigid flow statisctics have been obtained with the scripts "rigidFlowDataProcess.py" and "rigidFlowDataProcess_sequence.py". The files contain the rigid flow mean and variance for each pixel and (linear) depth quantile. The second script takes into consideration the mean and the variance for each sequence, the final result is the mean between the sequences.

# Motion Detection
## Detectron2 x SORT x Monodepth2

The main script is "motionDetectionKITTI.py". This script allows you to choose between 3 modes, each of them will create a KITTI_motion dataset according to the format written in the next part. Options are written in "mobile_options.py", letting you choose the mode and the parameters you want to use for each algorithm. These algorithms consist in different filters within "filter_model.py". Indeed, at each frame, our criterion will define wether one object is mobile or not. However, because of prediction errors and the accumulation of layers to calculate our criteria, there is a clear lack of consistency between each frame, hence the need to have a backpropagation that will allow us to consider mobile objects in the whole sequence. 

### Simple mode
The simple mode is a mode that will simply consider objects that were classified as mobile objects a fixed amount of time within the whole sequence.

### Ratio mode
The amout of time in this mode now depends on the existence time of the object in the sequence. Furthermore, we added the notion of lost objects, by considering the objects that were classified mobile in a frame as not mobile if they have not been considered mobile again in a fixed amount of time. 

### Score mode
In this mode we changed our criteria, so that they will give us a score between [0,1] instead of the binary output mobile or not mobile. We added to our filter the same ideas seen before, plus the idea of a mobile threshold, at which one object can reasonably be considered as mobile. 

## KITTI_motion dataset
The KITTI_motion dataset is the result of "motionDetectionKITTI.py" script. Depending on the mode you choose, the name of the dataset will be rather "KITTI_motion_ratio" or "KITTI_motion_score".

```cmd
python3 detectMovableObjectsKITTI.py --model_name mono+stereo_1024x320 --ext png --input /path/to/KITTI_raw --ckpt /path/to/detectron2/model.pth --dataset kitti --max_age 3
```

The script creates the dataset KITTI_motion, having the same file structure as KITTI_raw with masks of moving object for each image and instances data for each sequence.

```
KITTI_motion/
    | -- parameters.json
    | -- 2011_09_26/
    |       | -- 2011_09_26_drive_0001_sync/
    |       |       | -- image_03/
    |       |       |       | -- data/
    |       |       |       |       | -- 0000000000.png       
    |       |       |       |       ...
    |       |       |       |       | -- instance_data.json  
    ...

- parameters.json : dictionary of all the parameters used to obtain this dataset
          -> parameters["mode"]: the mode you chose between ["simple", "ratio", "score"]
          -> parameters["sort_age"] : max age before one tracklet in SORT is considered lost
          -> parameters["detectron2_threshold"] : threshold of detectron2 instance prediction
          -> parameters["monodepth2_model"] : the model used for monodepth2
          -> parameters["filter_age"] : Number of detected frames before one object should be considered mobile
          -> parameters["lost_age"] : Number of consecutive undetected frames before one object should be removed
          -> parameters["ratio"] : Ratio of detected frames before one object should be considered mobile
          -> parameters["alpha"] : Simple/ratio mode parameter
          -> parameters["beta"] : Simple/ratio mode parameter
          -> parameters["gamma"] : Simple/ratio mode parameter
          -> parameters["mobile_score_threshold"] : Threshold for which one object is considered as a mobile object
          -> parameters["delta"] : Score mode parameter
          -> parameters["omega"] : Score mode parameter
          -> parameters["sig1"] : Score mode parameter
          -> parameters["sig2"] : Score mode parameter
          -> parameters["sig3"] : Score mode parameter

- 0000000000.png : encoded mobile mask
          -> 0 corresponds to the background
          -> >0 corresponds to (1000*class+id), with class=[0=dynamic, 1=person,....] and id=[tracking id]

- instance_data.json : for each frame of the sequence, stores the id, the bounding box and the detectron2 score for each mobile instance detected
          -> instance_data["imgHeight"] : int, height of the image
          -> instance_data["imgWidth"].pred_masks : int width of the image
          -> instance_data["0000000000"] : List of mobile object dictionary
                  -> instance_data["0000000000"][i]["id"] : tracking id of the mobile object
                  -> instance_data["0000000000"][i]["bbox"] : bounding box of the object within the frame, format [x1,y1,x2,y2], upper right and bottom left                                                                         coordinates
                  -> instance_data["0000000000"][i]["score"] : detectron2 score of the object
```

# References



```latex
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```
https://github.com/nianticlabs/monodepth2

```latex
@misc{Cityscapes,
  doi = {10.48550/ARXIV.1604.01685},
  url = {https://arxiv.org/abs/1604.01685},
  author = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
  publisher = {arXiv},
  year = {2016},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```latex
@article{KITTI,
  author = {Hassan Alhaija and Siva Mustikovela and Lars Mescheder and Andreas Geiger and Carsten Rother},
  title = {Augmented Reality Meets Computer Vision: Efficient Data Generation for Urban Driving Scenes},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2018}
}
```

```latex
@inproceedings{Bewley2016_sort,
  author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
  title={Simple online and realtime tracking},
  year={2016},
  pages={3464-3468},
  keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
  doi={10.1109/ICIP.2016.7533003}
}
```

```latex
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
https://github.com/facebookresearch/detectron2
