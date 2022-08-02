from mobile_utils import *
from mobile_options import get_argparser
import simple_mode
import score_mode
import ratio_mode
import os 
import json

def main():
    args = get_argparser().parse_args()
    parameters = {}
    parameters["mode"] = args.mode
    parameters["sort_age"] = args.sort_age
    parameters["detectron2_threshold"] = args.detectron2_th
    parameters["monodepth2_model"] = args.model_name
    parameters["filter_age"] = args.filter_age
    parameters["lost_age"] = args.lost_age
    ### Simple mode
    if args.mode == "simple":
        parameters["alpha"] = args.alpha
        parameters["beta"] = args.beta
        parameters["gamma"] = args.gamma
        if not os.path.exists(args.input.replace("KITTI_raw", "KITTI_motion")):
            os.makedirs(args.input.replace("KITTI_raw", "KITTI_motion"))
        json_string = json.dumps(parameters, indent=4)
        with open(os.path.join(args.input.replace("KITTI_raw", "KITTI_motion"), "parameters.json"),'w') as file:
            file.write(json_string)
        simple_mode.main(args)

    ### Ratio mode
    if args.mode == "ratio":
        parameters["ratio"] = args.ratio
        parameters["alpha"] = args.alpha
        parameters["beta"] = args.beta
        parameters["gamma"] = args.gamma
        if not os.path.exists(args.input.replace("KITTI_raw", "KITTI_motion_ratio")):
            os.makedirs(args.input.replace("KITTI_raw", "KITTI_motion_ratio"))
        json_string = json.dumps(parameters, indent=4)
        with open(os.path.join(args.input.replace("KITTI_raw", "KITTI_motion_ratio"), "parameters.json"),'w') as file:
            file.write(json_string)
        ratio_mode.main(args)
    
    ### Score mode
    elif args.mode == "score":
        parameters["ratio"] = args.ratio
        parameters["mobile_score_threshold"] = args.mobile_score_th
        parameters["delta"] = args.delta
        parameters["omega"] = args.omega
        parameters["sigmoid1"] = args.sig1
        parameters["sigmoid2"] = args.sig2
        parameters["sigmoid3"] = args.sig3
        if not os.path.exists(args.input.replace("KITTI_raw", "KITTI_motion_score")):
            os.makedirs(args.input.replace("KITTI_raw", "KITTI_motion_score"))
        json_string = json.dumps(parameters, indent=4)
        with open(os.path.join(args.input.replace("KITTI_raw", "KITTI_motion_score"), "parameters.json"),'w') as file:
            file.write(json_string)
        score_mode.main(args)
    return

if __name__=="__main__":
    main()