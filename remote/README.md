
```
# Dataset Preparation
For one example dataset, i.e., the Marine Debris dataset, please refer to [SonarSAM](https://github.com/wangsssky/SonarSAM)

# Method

<img src="framework.png" width="800">

# Training
- Using box prompts
```
python train_SAM_box.py --config ./configs/sam_box.yaml
```
# Acknowledgment
Thanks for the awesome codes.
- Segment Anything Model: [SAM](https://github.com/facebookresearch/segment-anything)
- When SAM Meets Sonar Images: [SonarSAM](https://github.com/wangsssky/SonarSAM)
