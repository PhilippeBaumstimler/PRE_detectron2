# Python implementation of P² Quantile Estimator algorithm,
# inspired by : 
#       https://aakinshin.net/posts/p2-quantile-estimator/


import math
import os
import glob
import argparse
import numpy as np
import cv2 as cv
import PIL.Image as pil
import matplotlib.pyplot as plt
from skimage import exposure
from tqdm import tqdm
import torch
from torchvision import transforms

#monodepth2 import
from detectron2Monodepth2 import init_model_monodepth2, get_argparser
from layers import disp_to_depth


class P2QuantileEstimator:
    def __init__(self):
        self.n = [i for i in range(11)]
        self.dns = [i/10 for i in range(0,11)]
        self.ns = [x*10 for x in self.dns]
        self.q = []
        self.count = 0

    def addValue(self,x):
        if self.count<11:
            self.q.append(x)
            self.count+=1
            if self.count==5:
                self.q.sort()
        else:
            if x<self.q[0]:
                self.q[0]=x
                k=0
            elif x<self.q[1]:
                k=0
            elif x<self.q[2]:
                k=1
            elif x<self.q[3]:
                k=2
            elif x<self.q[4]:
                k=3
            elif x<self.q[5]:
                k=4
            elif x<self.q[6]:
                k=5
            elif x<self.q[7]:
                k=6
            elif x<self.q[8]:
                k=7
            elif x<self.q[9]:
                k=8
            elif x<self.q[10]:
                k=9
            else:
                self.q[10]=x
                k=9
            for i in range(k+1,11):
                self.n[i]+=1
            for i in range(11):
                self.ns[i]+=self.dns[i]
            for i in range(1,10):
                d = self.ns[i]-self.n[i]
                if(d>=1 and self.n[i+1]-self.n[i]>1) or (d<=-1 and self.n[i-1]-self.n[i]<-1):
                    dsign = int(math.copysign(1,d))
                    qs = self.Parabolic(i,dsign)
                    if(self.q[i-1]<qs and qs < self.q[i+1]):
                        self.q[i] = qs
                    else:
                        self.q[i] = self.Linear(i, dsign)
                    self.n[i] += dsign
            self.count+=1

    def Parabolic(self,i,sign):
        return self.q[i]+sign/(self.n[i+1]-self.n[i-1])*(
            (self.n[i]-self.n[i-1]+sign)*(self.q[i+1]-self.q[i])/(self.n[i+1]-self.n[i]) +
            (self.n[i+1]-self.n[i]-sign)*(self.q[i]-self.q[i-1])/(self.n[i]-self.n[i-1]))

    def Linear(self,i,sign):
        return self.q[i]+sign*(self.q[i+sign]-self.q[i])/(self.n[i+sign]-self.n[i])
    
    def GetQuantile(self,i):
        if self.count<11:
            print('Not enough values')
            return
        return self.q[i]

class LinearDepth:
    def __init__(self,n,max_depth):
        self.n = n
        self.max_depth= max_depth
        self.distribution = np.zeros(n)
        self.count = 0

    def addValue(self,x):
        ind = 1
        self.count +=1
        while(x> self.max_depth/self.n*ind and ind<self.n):
            ind +=1
        self.distribution[ind-1] +=1
    
    def getDistribution(self, normed=True):
        if normed:
            return self.distribution/self.count
        return self.distribution

def depthHist(args, depth):
    plt.figure(figsize=(12,3))
    plt.title('Distribution de profondeur')
    hist = np.histogram(depth, bins=100, density=True)
    plt.plot(hist[0])
    plt.savefig(os.path.join(args.output,"hist_depth.png"))
    plt.close()


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
    if os.path.isdir(args.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob.glob(os.path.join(args.input, '**/image_02/data/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(args.input):
        image_files.append(args.input)
    image_files.sort()

    output = os.path.join("KITTI_raw", "depth_distribution")
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
    ##### Setup quantile estimator
    q_estimator = P2QuantileEstimator()
    l_depth = LinearDepth(10, MAX_DEPTH)
    pixel_mean_depth = np.zeros((original_height, original_width))
    linear_depth_distribution = np.zeros(10)
    total_frame = 0
    # tot_depth=[]
    with torch.no_grad():
        for file in tqdm(image_files):
            total_frame+=1
            # Get image
            img_pil = pil.open(file).convert('RGB')

            # Depth prediction
            input_depth = img_pil.resize((feed_width, feed_height), pil.LANCZOS)
            input_depth = transforms.ToTensor()(input_depth).unsqueeze(0)
            input_depth = input_depth.to(device)
            features = depth_encoder(input_depth)
            depth_output = depth_decoder(features)
            disp = depth_output[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            _, depth = disp_to_depth(disp_resized_np, min_depth, max_depth)
            depth[depth < MIN_DEPTH] = MIN_DEPTH
            depth[depth > MAX_DEPTH] = MAX_DEPTH
            depth = [x for xs in depth for x in xs]
            # tot_depth = np.concatenate((tot_depth,depth))
            y = 0
            x = 0
            for value in depth:
                pixel_mean_depth[y//original_width,x%original_width] += value
                q_estimator.addValue(value)
                l_depth.addValue(value)
                y+=1
                x+=1
        pixel_mean_depth = pixel_mean_depth/total_frame
    # depthHist(args, tot_depth)
    cv.imwrite(os.path.join(output, "pixel_mean_depth.png"),pixel_mean_depth)
    np.save(os.path.join(output, "pixel_mean_depth.npy"),pixel_mean_depth)
    np.save(os.path.join(output,"linear_depth_distribution.npy"), l_depth.getDistribution())
    quantiles = [q_estimator.GetQuantile(i) for i in range(11)]
    np.save(os.path.join(output,"quantile_depth_distribution.npy"), np.array(quantiles))
    return

if __name__=="__main__":
    main()