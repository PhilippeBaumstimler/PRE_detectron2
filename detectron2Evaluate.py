# Script permettant d'entrainer sur une base de donné connue le réseau detectron2
# conceptualisé par le groupe facebook.

# L'objectif est d'étendre l'entrainement à la base de données KITTI qui ne fait pas
# partie des références connues de detectron2.

# Le code est inspiré du tutoriel colab fourni par facebook :
#       https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=FsePPpwZSmqt
# Et du script d'entrainement plain_train_net.py fourni par la documentation detectron2

import numpy as np
import os
import argparse
import torch
import logging
from collections import OrderedDict
from tqdm import tqdm

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
import detectron2CustomDataset as CustomDataset
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_writers
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format,
)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str, 
                        default= os.path.join(os.environ['CITYSCAPES_DATASET'],"leftImg8bit","val"),
                        help="chemin d'accès vers le fichier d'images ou l'image seule")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['kitti', 'cityscapes', 'kitti8'], help='Nom de la dataset')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="chemin modele pre-entraine")
    parser.add_argument("--output", default="output/evaluation",
                        help="repertoire de sauvegarde des prédictions de segmentation")
    
    ## Hyper-paramètres
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Taille du batch d'entrainement")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Valeur du learning rate")
    parser.add_argument("--max_iter", type=int, default=300,
                        help="Nombre d'itérations d'entrainement")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Nombre de classes d'objet à considérer")
    return parser


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "kitti_instance":
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def test(cfg, model, resume=False):
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)

        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        print("Evaluation results for {} in csv format:".format(dataset_name))
        print(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def main():
    args = get_argparser().parse_args()
    # Configuration du modèle  
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    if args.dataset == "cityscapes":
        os.environ['CITYSCAPES_DATASET']="/media/nicolas/data/cityscapes/"
        train_mode = "cityscapes_fine_instance_seg_train"
        test_mode = "cityscapes_fine_instance_seg_val"
    elif args.dataset == "kitti":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        #os.system('echo $CITYSCAPES_DATASET')
        train_mode = "kitti_seg_instance_train"
        test_mode = "kitti_seg_instance_val"
        cfg.INPUT.MASK_FORMAT = "bitmask"
        CustomDataset.create_kitti_dataset()
    elif args.dataset == "kitti8":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        #os.system('echo $CITYSCAPES_DATASET')
        train_mode = "kitti_seg_instance_train8"
        test_mode = "kitti_seg_instance_val8"
        cfg.INPUT.MASK_FORMAT = "bitmask"
        CustomDataset.create_kitti_dataset8()
    if args.num_classes != None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  
    if args.ckpt == None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  
    else:
        cfg.MODEL.WEIGHTS = args.ckpt 
    cfg.DATASETS.TRAIN = (train_mode,)
    cfg.DATASETS.TEST = (test_mode,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.BASE_LR = args.lr      
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.MAX_ITER = args.max_iter       
    cfg.OUTPUT_DIR = args.output
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.TEST.EVAL_PERIOD = 100000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Construction du modèle selon la configuration choisie
    model = build_model(cfg)
    test(cfg, model)
    return


if __name__ == "__main__":
    main()