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
import glob
from tqdm import tqdm
from PIL import Image
import torch
import matplotlib.pyplot as plt



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
import detectron2CustomDataset as CustomDataset

from sort.sort import *
from cityscapesscripts.helpers.labels import trainId2label

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
    parser.add_argument("--max_age", default=1, type=int,
                        help="Temps d'existance d'un tracker")
    parser.add_argument("--predict_on", action="store_true", default=False,
                        help="Prédit le déplacement d'une bbox qui a été perdue de vue")
    return parser


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


def main():
    args = get_argparser().parse_args()

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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    ##### Setup dataloader
    image_files = []
    if os.path.isdir(args.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob.glob(os.path.join(args.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(args.input):
        image_files.append(args.input)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    image_files.sort()

    ##### Setup tracker
    mot_tracker = Sort(max_age=args.max_age, min_hits=2, iou_threshold=0.2, predict_on=args.predict_on)
    total_frames = 0
    colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]
    
    ##### Setup video
    size = (1242,375)
    out = cv.VideoWriter(os.path.join(args.output,'sequence_age%d_predict%d.mp4'%(args.max_age, args.predict_on)), cv.VideoWriter_fourcc(*'mp4v'), 2, size)

    with torch.no_grad():
        for file in tqdm(image_files):
            total_frames +=1
            img_file = file.split("/")[-1]
            img_name = img_file.split(".")[0]
            img_cv = cv.imread(file)
            predictions = get_prediction(img_cv, cfg, model)
            img = np.array(img_cv)
            bboxes = [x.tolist() for x in predictions.pred_boxes]
            pred_masks = predictions.pred_masks.tolist()
            scores = predictions.scores.tolist()
            classes = predictions.pred_classes.tolist()
            detections=[]
            for bbox, score, classe, pred_mask in zip(bboxes, scores, classes, pred_masks):
                tmp = bbox   + [score]+ [classe] + [x for xs in pred_mask for x in xs]
                detections.append(tmp)
                detections.append(tmp)
            detections=np.array(detections)
            labels = _create_text_labels(classes, scores, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None))
            if len(detections)!=0:
                tracked_objects, unmatched = mot_tracker.update(detections)
            else:
                tracked_objects = mot_tracker.update()
            for x1,y1,x2,y2,cx,cy,obj_id,score,obj_cls,dict_mask in tracked_objects:
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                color = colors[int(obj_id) % len(colors)]
                cls = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)[int(obj_cls)]
                cv.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                cv.rectangle(img_cv, (x1, y1-10), (x1+len(labels)*4, y1), color, -1)
                cv.putText(img_cv, cls + "-" + str(int(obj_id)), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv.imwrite(os.path.join(args.output, img_name + "_tracking.png"), img_cv)
            img = cv.imread(os.path.join(args.output, img_name + "_tracking.png"))
            out.write(img)
    out.release()
    cv.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
