# Script permettant la prédiction de segmentation d'instance
# sur des images cityscapes au moyen du réseau detectron2 conceptualisé
# par Facebook

# En fournissant une image ou un fichier d'images en entrée, renvoie le résultat de
# la prédiction de segmentation d'instance sur l'image ou le fichier d'images dans 
# le dossier detectron2/output

# Le code est inspiré du tutoriel colab fourni par facebook :
#       https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=FsePPpwZSmqt

import numpy as np
import os
import argparse
import cv2 as cv
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch

import detectron2CustomDataset as CustomDataset

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask, _create_text_labels
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import detectron2.data.transforms as T

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str, required=True, 
                        default= os.path.join(os.environ['CITYSCAPES_DATASET'],"leftImg8bit","val"),
                        help="chemin d'accès vers le fichier d'images ou l'image seule")
    parser.add_argument("--type", type=str, default='instance',
                        choices=["instance", "semantic"], help = "type de segmentation - instance / semantique")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['kitti', 'cityscapes', 'kitti8'], help='Nom de la dataset')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="chemin modele pre-entraine")
    parser.add_argument("--num_classes", default=None, type=int,
                        help="Nombre de classes à prédire")
    parser.add_argument("--output", default="output/prediction",
                        help="repertoire de sauvegarde des prédictions de segmentation")
    return parser

def main():
    args = get_argparser().parse_args()

    setup_logger()

    #Setup modele
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    if args.dataset == "kitti":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        CustomDataset.create_kitti_dataset()
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
    elif args.dataset == "kitti8":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        CustomDataset.create_kitti_dataset8()
        cfg.DATASETS.TRAIN = ("kitti_seg_instance_train8",)
        cfg.INPUT.MASK_FORMAT = "bitmask"
    if args.ckpt == None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  
    else:
        #cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
        cfg.MODEL.WEIGHTS = args.ckpt
        cfg.INPUT.MIN_SIZE_TEST = 1024
        #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    if args.num_classes!=None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    # Setup dataloader
    image_files = []
    if os.path.isdir(args.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(args.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(args.input):
        image_files.append(args.input)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model.eval()
    with torch.no_grad():
        for file in tqdm(image_files):
            img_file = file.split("/")[-1]
            img_name = img_file.split(".")[0]

            img = cv.imread(file)
            if cfg.INPUT.FORMAT == "RGB":
                # whether the model expects BGR inputs or RGB
                img = img[:, :, ::-1]
            height, width = img.shape[:2]
            image = aug.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            outputs = model([inputs])[0]
            predictions = outputs["instances"].to("cpu")

            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
            scores = predictions.scores if predictions.has("scores") else None
            classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
            labels = _create_text_labels(classes, scores, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None))
            out = v.overlay_instances(
                masks=masks,
                boxes=predictions.pred_boxes,
                labels=labels,
                keypoints=None,
                assigned_colors=None,
                alpha=0.8,
            )
            out.save(os.path.join(args.output, img_name + ""))
    return

if __name__ == "__main__":
    main()
