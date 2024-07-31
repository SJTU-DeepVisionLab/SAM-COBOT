

## Installation 
1. Create a virtual environment `conda create -n SAM_PARSER python=3.10 -y` and activate it `conda activate SAM_PARSER`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/SJTU-DeepVisionLab/SAM-PARSER`
4. Enter the folder `cd medical` and run `pip install -e .`


### Data preparation and preprocessing

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it at `work_dir/SAM/sam_vit_b_01ec64.pth` .

Download the demo [dataset](https://zenodo.org/record/7860267) and unzip.

This dataset contains 50 abdomen CT scans and each scan contain an annotation mask with 13 organs. The names of the organ label are available at [MICCAI FLARE2022](https://flare22.grand-challenge.org/).
In this tutorial, we will fine-tune SAM for gallbladder segmentation.

Run pre-processing

```bash
python pre_CT.py -i path_to_image_folder -gt path_to_gt_folder -o path_to_output
```

- split dataset: 80% for training and 20% for testing
- image normalization
- pre-compute image embedding
- save the normalized images, ground truth masks, and image embedding as a `npz` file

We also provide a tutorial on 2D dataset (png format): finetune_and_inference_tutorial_2D_dataset.ipynb 

You can also train the model on the whole dataset. 
1) Download the training set ([GoogleDrive](https://drive.google.com/drive/folders/1pwpAkWPe6czxkATG9SmVV0TP62NZiKld?usp=share_link))

> Note: For the convenience of file sharing, we compress each image and mask pair in a `npz` file. The pre-computed image embedding is too large (require ~1 TB space). You can generate it with the following command

2) Pre-compute the image embedding and save the image embedding and ground truth as `.npy` files. 

```bash
python utils/precompute_img_embed.py -i path_to_train_folder -o ./data/Tr_npy
```

3) Train the model

```bash
python train.py
```

## Inference


Run

```bash
python Inference.py
```


## Acknowledgements
- We highly appreciate all the challenge organizers and dataset owners for providing the public dataset to the community. 
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Alexandre Bonnet for sharing this great [blog](https://encord.com/blog/learn-how-to-fine-tune-the-segment-anything-model-sam/)


