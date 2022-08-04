# Script permettant d'entrainer sur une base de donné connue le réseau detectron2
# conceptualisé par le groupe facebook.

# L'objectif est d'étendre l'entrainement à la base de données KITTI qui ne fait pas
# partie des références connues de detectron2.

# Le code est inspiré du tutoriel colab fourni par facebook :
#       https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=FsePPpwZSmqt
# Et du script d'entrainement plain_train_net.py fourni par la documentation detectron2


import os
import argparse
from collections import OrderedDict

import torch
import detectron2CustomDataset as CustomDataset
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2Evaluate import get_evaluator
setup_logger()

from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm

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

class ValidationLoss(HookBase):
    def __init__(self, cfg, DATASETS_VAL_NAME):# Add one more DATASETS_VAL_NAME Parameters （ Small changes ）
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = DATASETS_VAL_NAME##
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self,model,storage):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
    "val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            return losses_reduced, loss_dict_reduced
            # storage.put_scalars(total_val_loss=losses_reduced, 
            #                                      **loss_dict_reduced)


def test(cfg, model, resume=False):
    from detectron2.evaluation import inference_on_dataset
    from detectron2.data import build_detection_test_loader
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

def train(cfg, model, resume=False):
    val_loss = ValidationLoss(cfg, cfg.DATASETS.TEST)  ## Additional parameters 
    # model.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    # model._hooks = model._hooks[:-2] + model._hooks[-2:][::-1]
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
        checkpointer, 1000, max_iter=max_iter
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
            losses_reduced, loss_dict_reduced = val_loss.after_step(model,storage)
            storage.put_scalars(loss=losses, val_loss=losses_reduced)
            scheduler.step()
            # if (iteration+1)%cfg.TEST.EVAL_PERIOD==0:
            #     test(cfg,model)
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

def main():
    args = get_argparser().parse_args()
    if args.dataset == "cityscapes":
        os.environ['CITYSCAPES_DATASET']="/media/nicolas/data/cityscapes_pm/"
        #os.system('echo $CITYSCAPES_DATASET')
        train_mode = "cityscapes_fine_instance_seg_train"
        test_mode = "cityscapes_fine_instance_seg_val"
    elif args.dataset == "kitti":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        #os.system('echo $CITYSCAPES_DATASET')
        train_mode = "kitti_seg_instance_train"
        test_mode = "kitti_seg_instance_val"
        CustomDataset.create_kitti_dataset()
    elif args.dataset == "kitti8":
        os.environ['KITTI_SEG_DATASET']="/media/nicolas/data/KITTI_seg/"
        #os.system('echo $CITYSCAPES_DATASET')
        train_mode = "kitti_seg_instance_train8"
        test_mode = "kitti_seg_instance_val8"
        CustomDataset.create_kitti_dataset8()

    # Configuration du modèle
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    
    cfg = get_cfg()
    # cfg.MODEL.DEVICE="cpu"
    cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
    if args.dataset in ["kitti", "kitti8"]:
        cfg.INPUT.MASK_FORMAT = "bitmask"
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
    cfg.TEST.EVAL_PERIOD = 1000
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Construction du modèle selon la configuration choisie
    from detectron2.modeling import build_model
    model = build_model(cfg)

    #logger.info("Model:\n{}".format(model))
    train(cfg, model, args.resume)
    return


if __name__ == "__main__":
    main()
