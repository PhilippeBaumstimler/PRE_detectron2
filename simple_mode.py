import os
import glob
import json
import cv2 as cv
import numpy as np
from tqdm import tqdm
import PIL.Image as pil


import torch
from torchvision import transforms

#monodepth2 import
from layers import transformation_from_parameters, BackprojectDepth, Project3D, disp_to_depth

# import SORT algorithm
from sort.sort import Sort
from sort.sort import KalmanBoxTracker

from mobile_utils import init_model_detectron2, init_model_monodepth2, init_model_pose, img2TensorPose, get_prediction, getQuantileId,bbox_IoU
from filter_model import mobileObjects_simple

def main(args):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ##### Setup dataloader
    sequence_files = glob.glob(os.path.join(args.input, "**/image_03/data"), recursive=True)
    sequence_files.sort()

    ##### Setup model detectron2
    cfg, model = init_model_detectron2(args)
    model.eval()

    ##### Setup model depth
    depth_encoder, depth_decoder, feed_height, feed_width = init_model_monodepth2(args, device)
    depth_encoder.eval()
    depth_decoder.eval()
    min_depth = 0.1
    max_depth = 100.0

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    ##### Setup model pose
    pose_encoder, pose_decoder = init_model_pose(args, device)
    pose_encoder.eval()
    pose_decoder.eval()
    
    
    ##### Setup flow stats
    quantiles = np.load(os.path.join(args.statistics,"depth_distribution/quantile_depth_distribution.npy"))
    quantile_mean_flow = np.load(os.path.join(args.statistics,"rigid_flow_sequence/quantile_mean_flow.npy"))
    quantile_std_flow =np.sqrt(np.load(os.path.join(args.statistics,"rigid_flow_sequence/quantile_var_flow.npy")))
    tot_seq = len(sequence_files)
    with torch.no_grad():
        count = 0
        for sequence in sequence_files:
            total_frames = 0
            prev_pred={}

            # sequence_pred = []
            count+=1
            sequence_name = sequence.split('/')[-3]
            print("Processing sequence n°%d/%d, sequence name : %s"%(count,tot_seq,sequence_name))
            image_files = glob.glob(os.path.join(sequence, "**/*.png"), recursive=True)
            image_files.sort()

            if(len(image_files)>=2):
                img_height, img_width, _ = cv.imread(image_files[0]).shape

                # Camera projection
                K = np.array([[0.58, 0, 0.5, 0],
                            [0, 1.92, 0.5, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)
                K[0, :] *= img_width
                K[1, :] *= img_height
                inv_K = torch.from_numpy(np.linalg.pinv(K))[None].to(device)
                K = torch.from_numpy(K).to(device)

                ##### Setup Backprojection and Projection
                backproj = BackprojectDepth(1,img_height,img_width)
                backproj.to(device)
                proj3D = Project3D(1,img_height,img_width)
                proj3D.to(device)

                ##### Setup tracker
                KalmanBoxTracker.count = 0
                mot_tracker = Sort(img_height, img_width, max_age=args.sort_age, min_hits=1, iou_threshold=0.2, predict_on=args.predict_on)

                ##### Setup mobile tracker
                mobject = mobileObjects_simple(age=args.filter_age, threshold=0.9)

                for file in tqdm(image_files):
                    total_frames +=1
                    # Load image t and preprocess
                    img_file = file.split("/")[-1]
                    img_name = img_file.split(".")[0]

                    img_pil = pil.open(file).convert('RGB')
                    img_rgb = cv.imread(file)
                    
                    # Object detection
                    detectron_output = get_prediction(img_rgb, cfg, model)
                    bboxes = [x.tolist() for x in detectron_output.pred_boxes]
                    pred_masks = detectron_output.pred_masks.tolist()
                    scores = detectron_output.scores.tolist()
                    classes = detectron_output.pred_classes.tolist()

                    # Object tracking
                    detections=[]
                    for bbox, score, classe, pred_mask in zip(bboxes, scores, classes, pred_masks):
                        tmp = bbox   + [score]+ [classe] + [x for xs in pred_mask for x in xs]
                        detections.append(tmp)
                    detections=np.array(detections)
                    if len(detections)!=0:
                        tracked_objects, _ = mot_tracker.update(detections)
                    else:
                        tracked_objects = mot_tracker.update()

                    # Depth prediction
                    input_depth = img_pil.resize((feed_width, feed_height), pil.LANCZOS)
                    input_depth = transforms.ToTensor()(input_depth).unsqueeze(0)
                    input_depth = input_depth.to(device)
                    features = depth_encoder(input_depth)
                    depth_output = depth_decoder(features)
                    disp = depth_output[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (img_height, img_width), mode="bilinear", align_corners=False)
                    disp_resized_np = disp_resized.squeeze().cpu().numpy()

                    if total_frames>1:
                        # Pose prediction
                        prev_tensor = img2TensorPose(prev_pred["file"]).to(device)
                        curr_tensor = img2TensorPose(file).to(device)
                        all_color_aug = torch.cat([prev_tensor, curr_tensor], 1)
                        features = [pose_encoder(all_color_aug)]
                        axisangle, translation = pose_decoder(features)
                        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                        for object in prev_pred["tracked_objects"]:
                            if len(object)==10:
                                x1,y1,x2,y2,cx,cy,obj_id,_,obj_cls,dict_mask = object
                                # Average depth object
                                cx = min(max(0,int(cx)),img_width-1)
                                cy = min(max(0,int(cy)),img_height-1)
                                x1 = min(max(0,int(x1)),img_width-1)
                                y1 = min(max(0,int(y1)),img_height-1)
                                x2 = min(max(0,int(x2)),img_width-1)
                                y2 = min(max(0,int(y2)),img_height-1)
                                pred_mask = np.reshape(dict_mask["mask"]==1,(img_height,img_width))

                                _, prev_depth = disp_to_depth(prev_pred["disp"], min_depth, max_depth)
                                if(prev_depth[pred_mask].size>0):
                                    avg_depth = np.average(prev_depth[pred_mask]) 
                                    prev_depth[y1:y2, x1:x2] = avg_depth    
                                prev_depth[prev_depth < MIN_DEPTH] = MIN_DEPTH
                                prev_depth[prev_depth > MAX_DEPTH] = MAX_DEPTH     

                                # Backprojection and Projection
                                prev_depth =  torch.from_numpy(prev_depth).to(device)
                                cam_points = backproj(prev_depth, inv_K)
                                pix_coords = proj3D(cam_points, K, T)
                                
                                # Coord resize
                                pix_coords = pix_coords/2+0.5
                                pix_coords[...,0]*= (img_width-1)
                                pix_coords[...,1]*= (img_height-1)

                                # predicted bbox coordinates
                                pix_coords = pix_coords.squeeze().cpu().numpy()
                                px1 = min(max(0,int(pix_coords[y1,x1][0])),img_width-1)
                                py1 = min(max(0,int(pix_coords[y1,x1][1])),img_height-1)
                                px2 = min(max(0,int(pix_coords[y2,x2][0])),img_width-1)
                                py2 = min(max(0,int(pix_coords[y2,x2][1])),img_height-1)
                                pcx = int((pix_coords[y2,x2][0]-pix_coords[y1,x1][0])/2)
                                pcy = int((pix_coords[y2,x2][1]-pix_coords[y1,x1][1])/2)
                                dtrack = [x for x in tracked_objects if len(x)==10 and x[6]==obj_id]
                                if len(dtrack)>0:
                                    dx1,dy1,dx2,dy2,dcx,dcy,obj_id,score,_,_ = dtrack[0]
                                    dx1 = min(max(0,int(dx1)),img_width-1)
                                    dy1 = min(max(0,int(dy1)),img_height-1)
                                    dx2 = min(max(0,int(dx2)),img_width-1)
                                    dy2 = min(max(0,int(dy2)),img_height-1)
                                    dcx = int(dcx)
                                    dcy = int(dcy)
                                    idq = getQuantileId(prev_depth[y1,x1],quantiles)

                                    max_mean_flow = np.nanmax(quantile_mean_flow[idq,...])
                                    diff = np.linalg.norm([dcx-pcx, dcy-pcy])

                                    # GREEN box for detected bbox
                                    IoU = bbox_IoU([dx1,dy1,dx2,dy2],[px1,py1,px2,py2])
                                    if x1<=10 or x2>=img_width-11 or y1<=10 or y2>=img_height-11:
                                        if IoU < 1-args.alpha*quantile_mean_flow[idq,cy,cx]/max_mean_flow:
                                            mobject.update(int(obj_id),score)
                                    else:
                                        if 0 < (diff-args.beta*quantile_std_flow[idq,cy,cx])/max_mean_flow:
                                            mobject.update(int(obj_id),score)
                                        else:
                                            if IoU < (1+(diff-args.gamma*quantile_std_flow[idq,cy,cx])/max_mean_flow):
                                                mobject.update(int(obj_id),score)
                    prev_pred["file"] = file
                    prev_pred["img_name"] = img_name
                    prev_pred["tracked_objects"] = tracked_objects
                    prev_pred["bboxes"] = bboxes
                    prev_pred["pred_masks"] = pred_masks
                    prev_pred["classes"] = classes
                    prev_pred["disp"] = disp_resized_np
                    # sequence_pred.append(prev_pred.copy())
                print("Last filter...")
                mobile_ids = mobject.getMobileId()
                output = sequence.replace("KITTI_raw", "KITTI_motion")
                if not os.path.exists(output):
                    os.makedirs(output)
                ##### Setup tracker
                KalmanBoxTracker.count = 0
                mot_tracker = Sort(img_height, img_width, max_age=args.sort_age, min_hits=1, iou_threshold=0.2, predict_on=args.predict_on)
                jsonDict = {}
                jsonDict["imgHeight"] = img_height
                jsonDict["imgWidth"] = img_width
                if(len(image_files)>=2):
                    for file in tqdm(image_files):
                        mask = np.zeros((img_height,img_width))
                        # Load image t and preprocess
                        img_file = file.split("/")[-1]
                        img_name = img_file.split(".")[0]
                        img_rgb = cv.imread(file)  
                        img_height, img_width, _ = img_rgb.shape

                        # Object detection
                        detectron_output = get_prediction(img_rgb, cfg, model)
                        bboxes = [x.tolist() for x in detectron_output.pred_boxes]
                        pred_masks = detectron_output.pred_masks.tolist()
                        scores = detectron_output.scores.tolist()
                        classes = detectron_output.pred_classes.tolist()

                        # Object tracking
                        detections=[]
                        for bbox, score, classe, pred_mask in zip(bboxes, scores, classes, pred_masks):
                            tmp = bbox   + [score]+ [classe] + [x for xs in pred_mask for x in xs]
                            detections.append(tmp)
                        detections=np.array(detections)
                        if len(detections)!=0:
                            tracked_objects, _ = mot_tracker.update(detections)
                        else:
                            tracked_objects = mot_tracker.update()
                        objects = []
                        for object in tracked_objects:
                            if len(object)==10:
                                x1,y1,x2,y2,_,_,obj_id,score,obj_cls,dict_mask = object
                                dict_object ={}
                                if obj_id in mobile_ids:
                                    instance_mask = np.reshape(dict_mask["mask"],(img_height,img_width))
                                    mask = np.maximum(mask, instance_mask*(obj_cls*1000+obj_id))
                                    dict_object["id"] = obj_id
                                    dict_object["bbox"] = [x1,y1,x2,y2]
                                    dict_object["score"] = score
                                    objects.append(dict_object)
                        jsonDict[img_name] = objects
                        mobile_instance = pil.fromarray(np.uint32(mask))
                        mobile_instance.save(os.path.join(output,img_name +".png" ))
                json_string = json.dumps(jsonDict, indent=4)
                with open(os.path.join(output, "instance_data.json"),'w') as outfile:
                    outfile.write(json_string)

    return


