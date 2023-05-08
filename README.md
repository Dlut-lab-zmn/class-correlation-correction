# Class Correlation Correction for Unbiased Scene Graph Generation

## Introduction
The long-tail distribution in the scene graph generation (SGG) task has spurred immense interest in unbiased SGG. However, current state-of-the-art debiasing techniques extract statistics from the independent category, while ignoring the correlation between categories. To address this issue, we propose a simple but effective method, class correlation correction, to aggregate dependency knowledge among various classes.
Specifically, given biased predictions, two kinds of debiasing transformations are developed employing the class correlation aware label to recover the unbiased estimates. 
We also propose to retrain SGG models with the biasing transformations adapted to the biased data distribution. 
To evaluate the debiasing performance, several biased datasets are constructed from CIFAR-10 and Fashion-MNIST in addition to the widely employed SGG dataset.
Extensive experiments on these datasets and multiple evaluation metrics demonstrate the efficacy of the proposed method.


### Baseline (Backbone Only)
- Training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR output/motif-precls-exmp SOLVER.PRE_VAL True
```

- Validation
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 48 DTYPE "float16" GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR output/motif-precls-exmp
```

### F3C (or B3C)

- Training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False  MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR output/vctree-sgdet-exmp-balanced_norm LOG_TB True SOLVER.PRE_VAL True MODEL.BALANCED_NORM True
```




- Validation
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 48 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 1000 SOLVER.CHECKPOINT_PERIOD 1000 GLOVE_DIR glove MODEL.PRETRAINED_DETECTOR_CKPT checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR output/motif-precls-exmp-balanced_norm LOG_TB True SOLVER.PRE_VAL True MODEL.BALANCED_NORM True
```

## Citation
If you think this code is helpful, please cite the following papers
```
@inproceedings{chiou2021recovering,
    title={Recovering the Unbiased Scene Graphs from the Biased Ones},
    author={Chiou, Meng-Jiun and Ding, Henghui and Yan, Hanshu and Wang, Changhu and Zimmermann, Roger and Feng, Jiashi},
    booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
    pages={1581–-1590},
    year={2021}
}
Ours to be appear...
```

