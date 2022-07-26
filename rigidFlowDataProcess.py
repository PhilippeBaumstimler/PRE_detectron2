import os
import glob
import math
from shutil import register_unpack_format
import torch
import argparse
import numpy as np
import PIL.Image as pil
from tqdm import tqdm
from torchvision import transforms
#monodepth2 import
from detectron2Monodepth2 import init_model_monodepth2, init_model_pose, get_argparser, img2TensorPose
from layers import disp_to_depth, transformation_from_parameters, BackprojectDepth, Project3D


def main():
    args = get_argparser().parse_args()
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    ##### Setup dataloader
    original_height = 375
    original_width = 1242
    image_files = []

    sequence_files = glob.glob(os.path.join(args.input, "**/image_02/data"), recursive=True)
    sequence_files.sort()

    output = os.path.join("KITTI_raw", "rigid_flow_distribution2")
    if not os.path.exists(output):
        os.makedirs(output)

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
    K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
    K[0, :] *= original_width
    K[1, :] *= original_height
    inv_K = torch.from_numpy(np.linalg.pinv(K))[None].to(device)
    K = torch.from_numpy(K).to(device)

    ##### Setup Backprojection and Projection
    backproj = BackprojectDepth(1,original_height,original_width)
    backproj.to(device)
    proj3D = Project3D(1,original_height,original_width)
    proj3D.to(device)
    
    ##### Setup Flow statistics
    quantiles = np.load("KITTI_raw/depth_distribution/quantile_depth_distribution.npy")

    tot_seq = len(sequence_files)
    global_flow = np.zeros((original_height,original_width))
    global_flow_square = np.zeros((original_height,original_width))
    quantile_flow_total = np.zeros((10,original_height, original_width))
    quantile_flow_square = np.zeros((10,original_height, original_width))
    linear_flow_total = np.zeros((10,original_height, original_width))
    linear_flow_square = np.zeros((10,original_height, original_width))
    quantile_flow_count = np.zeros((10,original_height, original_width))
    linear_flow_count = np.zeros((10,original_height, original_width))
    with torch.no_grad():
        n_seq = 0
        tot_frame=0
        for sequence in sequence_files:
            n_seq+=1
            sequence_name = sequence.split('/')[-3]
            print("Processing sequence n°%d/%d, sequence name : %s"%(n_seq,tot_seq,sequence_name))
            image_files = glob.glob(os.path.join(sequence, "**/*.png"), recursive=True)
            image_files.sort()
            frame = 0
            prev_pred = {}
            if(len(image_files)>=2):
                for file in tqdm(image_files):
                    frame +=1
                    img_pil = pil.open(file).convert('RGB')
                    img_file = file.split("/")[-1]
                    img_name = img_file.split(".")[0]
                    # Depth prediction
                    input_depth = img_pil.resize((feed_width, feed_height))
                    input_depth = transforms.ToTensor()(input_depth).unsqueeze(0)
                    input_depth = input_depth.to(device)
                    features = depth_encoder(input_depth)
                    depth_output = depth_decoder(features)
                    disp = depth_output[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (original_height, original_width), mode="bilinear", align_corners=False)
                    _, depth = disp_to_depth(disp_resized, min_depth, max_depth)
                    depth[depth < MIN_DEPTH] = MIN_DEPTH
                    depth[depth > MAX_DEPTH] = MAX_DEPTH
                    depth = depth.squeeze()
                    if frame >1:
                        # Pose estimation
                        prev_tensor = img2TensorPose(prev_pred["file"]).to(device)
                        curr_tensor = img2TensorPose(file).to(device)
                        all_color_aug = torch.cat([prev_tensor, curr_tensor], 1)
                        features = [pose_encoder(all_color_aug)]
                        axisangle, translation = pose_decoder(features)
                        T = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
                        
                        # Rigid flow computation
                        prev_depth = prev_pred["depth"]
                        cam_points = backproj(prev_depth, inv_K)
                        pix_coords = proj3D(cam_points, K, T)
                        pix_coords = pix_coords/2+0.5
                        pix_coords[...,0]*= (original_width-1)
                        pix_coords[...,1]*= (original_height-1)
                        rigid_flow_map = pix_coords - backproj.back_coords.to(device)
                        rigid_flow_map = rigid_flow_map.cpu().numpy().squeeze()

                        # Rigid speed
                        rigid_flow_norm = np.linalg.norm(rigid_flow_map, axis=2)
                        prev_depth = prev_depth.cpu().numpy()

                        for i in range(0,len(quantiles)-1):
                            # Quantile depth distribution
                            mask_quantile = (quantiles[i]<prev_depth)&(prev_depth<=quantiles[i+1])
                            quantile_flow_total[i,...] += rigid_flow_norm*mask_quantile
                            quantile_flow_square[i,...] += np.square(rigid_flow_norm*mask_quantile)
                            # Linear depth distribution
                            mask_linear = (MAX_DEPTH/10*i<prev_depth)&(prev_depth<=MAX_DEPTH/10*i+1)
                            linear_flow_total[i,...] += rigid_flow_norm*mask_linear
                            linear_flow_square[i,...] += np.square(rigid_flow_norm*mask_linear)
                            #Count
                            quantile_flow_count[i,...] += mask_quantile
                            linear_flow_count[i,...] += mask_linear

                        global_flow+=rigid_flow_norm
                        global_flow_square += np.square(rigid_flow_norm)
                        
                    prev_pred["file"] = file
                    prev_pred["depth"] = depth
                tot_frame+=frame
    np.save(os.path.join(output, "global_mean_flow.npy"),global_flow/tot_frame)
    np.save(os.path.join(output, "global_var_flow.npy"),global_flow_square/tot_frame-np.square(global_flow/tot_frame))
    quantile_flow_total = np.divide(quantile_flow_total,quantile_flow_count, out = np.zeros_like(quantile_flow_count),where=quantile_flow_count!=0)
    quantile_flow_square = np.divide(quantile_flow_square,quantile_flow_count, out = np.zeros_like(quantile_flow_count),where=quantile_flow_count!=0)
    np.save(os.path.join(output, "quantile_mean_flow.npy"),quantile_flow_total)
    np.save(os.path.join(output, "quantile_var_flow.npy"),quantile_flow_square - np.square(quantile_flow_total))

    for i in range(10):
        linear_flow_total[i,...] = np.divide(linear_flow_total[i,...],linear_flow_count, out = np.zeros_like(linear_flow_count),where=linear_flow_count!=0)
        linear_flow_square[i,...] = np.divide(linear_flow_square[i,...],linear_flow_count, out = np.zeros_like(linear_flow_count),where=linear_flow_count!=0)
    np.save(os.path.join(output, "linear_mean_flow.npy"),linear_flow_total)
    np.save(os.path.join(output, "linear_var_flow.npy"),linear_flow_square - np.square(linear_flow_total))
    
    np.save(os.path.join(output, "tot_frame.npy"), [tot_frame])
    np.save(os.path.join(output, "quantile_flow_count.npy"), quantile_flow_count)
    np.save(os.path.join(output, "linear_flow_count.npy"), linear_flow_count)
    return

if __name__=="__main__":
    main()