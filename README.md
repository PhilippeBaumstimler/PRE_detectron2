# PRE_detectron2

Install Cytiscapes label script helper
```cmd
python -m pip install cityscapesscripts
```

Install detectron2 repository

```cmd
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)
```

Test script detectron2

```cmd
python detectron2.py --input /path/to/a/video/sequence/or/a/single/image/from/KITTI --dataset kitti --ckpt /path/to/model.pth
```

