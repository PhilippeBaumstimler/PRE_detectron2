import argparse

def get_argparser():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument("--input",type=str, default="/media/nicolas/data/KITTI_raw", 
                        help="Path to KITTI_raw")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["simple","ratio","score"])
    parser.add_argument("--statistics", type=str, default="KITTI_raw",
                        help="Path to statisctics npy")
    # Detectron2 options
    parser.add_argument("--dataset", type=str, default='kitti',
                        choices=['kitti', 'cityscapes', 'kitti8'], help='Dataset name')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="Path to checkpoint")
    parser.add_argument("--detectron2_th", type=float, default=0.3,
                         help="Detectron2 instance threshold")


    # SORT Tracking algorithm options
    parser.add_argument("--sort_age", default=3, type=int,
                        help="Tracklets max age")
    parser.add_argument("--predict_on", action="store_true", default=False,
                        help="Predicts the position of the tracklet if disapeared")

    # Monodepth2 options
    parser.add_argument('--model_name', default="mono+stereo_1024x320", type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', default="png",type=str,
                        help='image extension to search for in folder')
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    parser.add_argument('--num_layers', type=int, default="18",
                        choices=[18, 34, 50, 101, 152], help='Nombre de couche resnet')

    #Last filter options
    parser.add_argument('--filter_age', type=float, default=3,
                        help='Number of detected frames before one object should be considered mobile')
    parser.add_argument('--ratio', type=float, default=0.25,
                        help='Ratio of detected frames before one object should be considered mobile')
    parser.add_argument('--lost_age', type=int, default=5,
                        help='Number of consecutive undetected frames before one object should be removed')
    ## Simple/Ratio mode parameters
    parser.add_argument('--alpha', type=float, default=12,
                        help='Simple/ratio mode parameter')
    parser.add_argument('--beta', type=float, default=1.5,
                        help='Simple/ratio mode parameter')
    parser.add_argument('--gamma', type=float, default=1.5,
                        help='Simple/ratio mode parameter')
    ## Score mode parameters
    parser.add_argument('--mobile_score_th', type=float, default=0.7,
                        help='Mobile score threshold')
    parser.add_argument('--delta', type=float, default=0.4,
                        help='Score mode parameter')
    parser.add_argument('--omega', type=float, default=1,
                        help='Score mode parameter')
    parser.add_argument('--sig1', type=float, default=12,
                        help='Score mode parameter')
    parser.add_argument('--sig2', type=float, default=3,
                        help='Score mode parameter')
    parser.add_argument('--sig3', type=float, default=6,
                        help='Score mode parameter')
    return parser