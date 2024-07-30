# SAM-COBOT
The official implementation of CVPR 2024 paper:"Parameter Efficient Fine-tuning via Cross Block Orchestration for Segment Anything Model". If this project is helpful to your research, please consider citing our paper [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Peng_Parameter_Efficient_Fine-tuning_via_Cross_Block_Orchestration_for_Segment_Anything_CVPR_2024_paper.pdf).
```
@inproceedings{peng2024parameter,
  title={Parameter efficient fine-tuning via cross block orchestration for segment anything model},
  author={Peng, Zelin and Xu, Zhengqin and Zeng, Zhilin and Xie, Lingxi and Tian, Qi and Shen, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3743--3752},
  year={2024}
}
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
