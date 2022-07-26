# Script permettant d'entrainer sur une base de donné connue le réseau detectron2
# conceptualisé par le groupe facebook.

# L'objectif est d'étendre l'entrainement à la base de données KITTI qui ne fait pas
# partie des références connues de detectron2.

# Le code est inspiré du tutoriel colab fourni par facebook :
#       https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=FsePPpwZSmqt
# Et du script d'entrainement plain_train_net.py fourni par la documentation detectron2

from dataclasses import dataclass
from xmlrpc.client import boolean
import os
import argparse
import torch
import detectron2CustomDataset as CustomDataset
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
setup_logger()

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['kitti', 'cityscapes', 'kitti8'], help='Nom de la dataset')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="chemin modele pre-entraine")
    parser.add_argument("--output", default="output/train",
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
    parser.add_argument("--resume", action='store_true', default=False,
                        help="Reprend à la dernière itération")
    return parser


def train(cfg, model, resume=False):
    model.train()
    from detectron2.solver import build_lr_scheduler, build_optimizer
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    #Chargement du checkpoint et récupération des données
    from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
    from detectron2.engine import default_writers
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = default_writers(cfg.OUTPUT_DIR, max_iter)

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    from detectron2.data import build_detection_train_loader
    from detectron2.utils.events import EventStorage
    data_loader = build_detection_train_loader(cfg)
    #logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            storage.put_scalar("loss", losses)
            scheduler.step()

            # if (
            #     cfg.TEST.EVAL_PERIOD > 0
            #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
            #     and iteration != max_iter - 1
            # ):
            #     do_test(cfg, model)
            if (iteration+1)%20==0 :
                print("Itération :{}/{}, loss :{}".format(iteration+1, max_iter, losses))
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                #logger.info("Itération :{}/{}, loss :{}".format(iteration, max_iter, losses))
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def main():
    args = get_argparser().parse_args()
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

    # Configuration du modèle
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    if args.dataset in ["kitti", "kitti8"]:
        cfg.INPUT.MASK_FORMAT = "bitmask"
    if args.num_classes != None:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  
    if args.ckpt == None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")  
    else:
        cfg.MODEL.WEIGHTS = args.ckpt 
    cfg.DATASETS.TRAIN = (train_mode)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.BASE_LR = args.lr      
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.MAX_ITER = args.max_iter       
    cfg.OUTPUT_DIR = args.output
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.TEST.EVAL_PERIOD = 10000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Construction du modèle selon la configuration choisie
    from detectron2.modeling import build_model
    model = build_model(cfg)

    #logger.info("Model:\n{}".format(model))
    train(cfg, model, args.resume)
    return


if __name__ == "__main__":
    main()
