import os
import sys
import glob
import math
import shutil
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
import PIL.Image as pil
from PIL import ImageOps
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch


# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T
import detectron2CustomDataset as CustomDataset


def get_argparser():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument("--input",type=str, required=True, 
                        help="chemin d'accès vers fichier d'images")
    parser.add_argument("--output", default="output/prediction",
                        help="repertoire de sauvegarde des prédictions de segmentation")

    # Detectron2 options
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['kitti', 'cityscapes', 'kitti8'], help='Nom de la dataset')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="chemin modele pre-entraine")
    return parser

def init_model_detectron2(dataset, ckpt=None):
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
        cfg.MODEL.WEIGHTS = ckpt
        cfg.INPUT.MIN_SIZE_TEST = 1024
        #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    return cfg, model

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

if __name__=="__main__":
    args = get_argparser().parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ##### Setup model detectron2
    cfg, model = init_model_detectron2(args.dataset, args.ckpt)
    model.eval()

    ##### Setup dataloader
    original_height = 375
    original_width = 1242
    image_files = []
    if os.path.isdir(args.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob.glob(os.path.join(args.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(args.input):
        image_files.append(args.input)
    image_files.sort()

    with torch.no_grad():
        for file in image_files:
            img_rgb = cv.imread(file)

            #Output object detection
            detectron_output = get_prediction(img_rgb, cfg, model)
            bboxes = detectron_output.pred_boxes # Tensor (N,4) bouding boxes for each detected instance, format (x1,y1,x2,y2)
            pred_masks = detectron_output.pred_masks # Tensor (N,H,W) masks for each detected instance
            scores = detectron_output.scores # Tensor of N confidence score for each detected instance
            classes = detectron_output.pred_classes # Tensor of N labels for each detected instance
            # cls = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)[int(obj_cls)]