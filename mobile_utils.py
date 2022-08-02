import os
import cv2 as cv
import numpy as np

import torch
from torchvision import transforms

#monodepth2 import

import networks
from utils import download_model_if_doesnt_exist, readlines

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import detectron2CustomDataset as CustomDataset

def get_prediction(img, cfg, model):
    aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    if cfg.INPUT.FORMAT == "RGB":
        # whether the model expects BGR inputs or RGB
        img = img[:, :, ::-1]
    height, width = img.shape[:2]
    image = aug.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}
    outputs = model([inputs])[0]
    return outputs["instances"].to("cpu")




def init_model_detectron2(args):
    setup_logger()

    ##### Setup model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    if args.dataset == "kitti":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        CustomDataset.create_kitti_dataset()
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    elif args.dataset == "kitti8":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        CustomDataset.create_kitti_dataset8()
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train8",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    if args.ckpt == None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  
    else:
        #cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.WEIGHTS = args.ckpt
        cfg.INPUT.MIN_SIZE_TEST = 1024
        #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return cfg, model




def init_model_monodepth2(args, device):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading depth model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained depth encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)

    print("   Loading pretrained depth decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)

    return encoder, depth_decoder, feed_height, feed_width


def init_model_pose(args, device):
    model_path = os.path.join("models", args.model_name)
    print("-> Loading pose model from ", model_path)
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained pose encoder")
    pose_encoder = networks.ResnetEncoder(args.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path,map_location=device))

    print("   Loading pretrained pose decoder")
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path,map_location=device))

    pose_encoder.to(device)
    pose_decoder.to(device)

    return pose_encoder, pose_decoder

def img2TensorPose(file):
    img = cv.imread(file)
    x = transforms.ToTensor()(img)
    x = transforms.Resize((320,1024))(x)
    x = torch.unsqueeze(x, dim=0)
    return x

def bbox_IoU(bboxA, bboxB):
    #Intersection coordinates
    x1_int = max(bboxA[0],bboxB[0])  
    y1_int = max(bboxA[1],bboxB[1])
    x2_int = min(bboxA[2],bboxB[2])  
    y2_int = min(bboxA[3],bboxB[3])

    #Areas
    area_int = max(0,x2_int-x1_int+1)*max(0, y2_int-y1_int+1)
    area_bboxA = (bboxA[2]-bboxA[0]+1)*(bboxA[3]-bboxA[1]+1)
    area_bboxB = (bboxB[2]-bboxB[0]+1)*(bboxB[3]-bboxB[1]+1)
    #Intersection over Union
    return area_int/float(area_bboxA + area_bboxB - area_int)

def getQuantileId(depth, quantiles):
    ind=0
    for quantile in quantiles[1::]:
        if depth<=quantile:
            return ind
        ind+=1        

def sigmoid(r,x):
    sig = np.where(x < 0, np.exp(r*x)/(1 + np.exp(r*x)), 1/(1 + np.exp(-r*x)))
    return sig
