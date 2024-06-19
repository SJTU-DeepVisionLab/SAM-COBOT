import torch
from torch.utils.data import DataLoader

import numpy as np

import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
from utils.config import *
# from model.model_proxy import Proxy, ModelWithLoss
from model.model_proxy_SAM_box import SonarSAM, ModelWithLoss
from model.loss_functions import compute_dice_accuracy, compute_multilabel_dice_accuracy, compute_multilabel_IoU
from dataloader.data_loader import SAM_DebrisDataset, collate_fn_seq_box_seg_pair
from utils.utils import rand_seed
from model.segment_anything.utils.transforms import ResizeLongestSide


label_list = ["Background", "Bottle", "Can", "Chain", "Drink-carton", "Hook", 
              "Propeller", "Shampoo-bottle", "Standing-bottle", "Tire", "Valve", 
              "Wall"]



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))   


# def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
#     model.eval()
#     ious = AverageMeter()
#     f1_scores = AverageMeter()

#     with torch.no_grad():
#         for iter, data in enumerate(val_dataloader):
#             images, bboxes, gt_masks = data
#             num_images = images.size(0)
#             pred_masks, _ = model(images, bboxes)
#             pred_masks_for_vis = torch.zeros(1024,1024).cuda()
#             for pred_mask, gt_mask in zip(pred_masks, gt_masks):
#                 pred_mask = torch.sigmoid(pred_mask)
#                 #ipdb.set_trace()
#                 pred_mask = (pred_mask > 0.5)
#                 for i in range(pred_mask.shape[0]):
#                     pred_masks_for_vis += pred_mask[i]
            
#             fig, ax = plt.subplots(1)

#             pred_masks_for_vis = pred_masks_for_vis.cpu().numpy().astype(np.uint8)
#             # Display original image
#             ax.imshow(images.squeeze(0).cpu().numpy().transpose((1,2,0)))


#             #Display the segmentation mask
#             pred_mask[pred_mask>1] = 1
#             show_mask(pred_masks_for_vis, ax)
#             #ipdb.set_trace()
#             for i in range(bboxes[0].shape[0]):
#             # Display the bounding box
#                 show_box(bboxes[0][i].cpu().numpy(), ax)
#             #ax.axis("off")
#             #ax.set_xlim(0, images.shape[1])
#             #ax.set_ylim(images.shape[0], 0)      

#             plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
#             plt.margins(0,0)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())      
#             # Save the figure
#             save_path = os.path.join("./output_vis_CVPR2024/Adapter", f"result_{iter}.png")
#             os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             plt.savefig(save_path, bbox_inches='tight')
#             plt.close(fig)


def evaluate_vis(net, val_loader, device, opt):
    dice_ = [[], [], [], [], [], [], [], [], [], [], [], []]
    net.eval()
    with torch.no_grad():
        for val_step, (images, box_mask_pairs) in enumerate(tqdm(val_loader)):        
            images = images.to(device)
            
            boxes_batch = []
            masks_batch = []
            box_mask_pairs = box_mask_pairs[0]
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks_item = box_mask_pairs[idx]['masks']
                boxes_xyxy = []
                masks = []
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    box = box[:4]                   
                    boxes_xyxy.append(box)            
                    masks.append(masks_item[i].cuda())
                boxes_xyxy = np.array(boxes_xyxy)   
                H, W = images.shape[-2], images.shape[-1]
                sam_trans = ResizeLongestSide(net.sam.image_encoder.img_size)
                boxes_trans = sam_trans.apply_boxes(boxes_xyxy, (H, W))
                boxes_trans = torch.as_tensor(boxes_trans, dtype=torch.float, device=device)
            
                boxes_batch.append(boxes_trans)
                masks_batch.append(masks)               
                
            predictions = net.forward(images, boxes_batch)
            start_x = int(opt.INPUT_SIZE / 3.0) // 2
            end_x = opt.INPUT_SIZE -1 -start_x
            # masks = masks[:, :, :, start_x:end_x].contiguous()
            # predictions = predictions[:, :, :, start_x:end_x].contiguous()

            # eval metric
            pred_masks_for_vis = torch.zeros(H, W).cuda()                                        
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks = masks_batch[idx]
                pred_masks = predictions[idx]
                #ipdb.set_trace()
                for i in range(len(boxes_item)):
                    pred_masks_for_vis += (torch.sigmoid(pred_masks[i])>0.5).contiguous()
            
            fig, ax = plt.subplots(1)

            pred_masks_for_vis = pred_masks_for_vis.cpu().numpy().astype(np.uint8)
            # Display original image
            ax.imshow(images.squeeze(0).cpu().numpy().transpose((1,2,0)))


            #Display the segmentation mask
            #pred_mask[pred_mask>1] = 1
            show_mask(pred_masks_for_vis, ax)
            #ipdb.set_trace()
            for i in range(boxes_batch[0].shape[0]):
            # Display the bounding box
                show_box(boxes_batch[0][i].cpu().numpy(), ax)
            #ax.axis("off")
            #ax.set_xlim(0, images.shape[1])
            #ax.set_ylim(images.shape[0], 0)      

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())      
            # Save the figure
            save_path = os.path.join("./output_vis_CVPR2024/Adapter_relation", f"result_{val_step}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # if fig.dtype == np.float32 or fig.dtype == np.float64:
            #     fig = np.clip(fig, 0.0, 1.0)

            # # Convert to uint8 if the image is integer type
            # if fig.dtype != np.uint8:
            #     fig = np.clip(fig, 0, 255).astype(np.uint8)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)



def evaluate_image(net, val_loader, device, opt):
    dice_ = [[], [], [], [], [], [], [], [], [], [], [], []]
    net.eval()
    with torch.no_grad():
        for val_step, (images, box_mask_pairs) in enumerate(tqdm(val_loader)):        
            images = images.to(device)
            
            boxes_batch = []
            masks_batch = []
            box_mask_pairs = box_mask_pairs[0]
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks_item = box_mask_pairs[idx]['masks']
                boxes_xyxy = []
                masks = []
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    box = box[:4]                   
                    boxes_xyxy.append(box)            
                    masks.append(masks_item[i].cuda())
                boxes_xyxy = np.array(boxes_xyxy)   
                H, W = images.shape[-2], images.shape[-1]
                sam_trans = ResizeLongestSide(net.sam.image_encoder.img_size)
                boxes_trans = sam_trans.apply_boxes(boxes_xyxy, (H, W))
                boxes_trans = torch.as_tensor(boxes_trans, dtype=torch.float, device=device)
            
                boxes_batch.append(boxes_trans)
                masks_batch.append(masks)               
                
            predictions = net.forward(images, boxes_batch)
            start_x = int(opt.INPUT_SIZE / 3.0) // 2
            end_x = opt.INPUT_SIZE -1 -start_x
            # masks = masks[:, :, :, start_x:end_x].contiguous()
            # predictions = predictions[:, :, :, start_x:end_x].contiguous()

            # eval metric
            pred_masks_for_vis = torch.zeros(H, W).cuda()                                        
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks = masks_batch[idx]
                pred_masks = predictions[idx]
                #ipdb.set_trace()
                for i in range(len(boxes_item)):
                    pred_masks_for_vis += (torch.sigmoid(pred_masks[i])>0.5).contiguous()
            
            fig, ax = plt.subplots(1)

            pred_masks_for_vis = pred_masks_for_vis.cpu().numpy().astype(np.uint8)
            # Display original image
            ax.imshow(images.squeeze(0).cpu().numpy().transpose((1,2,0)))


            #Display the segmentation mask
            #pred_mask[pred_mask>1] = 1
            # show_mask(pred_masks_for_vis, ax)
            # #ipdb.set_trace()
            # for i in range(boxes_batch[0].shape[0]):
            # # Display the bounding box
            #     show_box(boxes_batch[0][i].cpu().numpy(), ax)
            #ax.axis("off")
            #ax.set_xlim(0, images.shape[1])
            #ax.set_ylim(images.shape[0], 0)      

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())      
            # Save the figure
            save_path = os.path.join("./output_vis_CVPR2024/image", f"result_{val_step}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # if fig.dtype == np.float32 or fig.dtype == np.float64:
            #     fig = np.clip(fig, 0.0, 1.0)

            # # Convert to uint8 if the image is integer type
            # if fig.dtype != np.uint8:
            #     fig = np.clip(fig, 0, 255).astype(np.uint8)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)


def evaluate_gt(net, val_loader, device, opt):
    dice_ = [[], [], [], [], [], [], [], [], [], [], [], []]
    net.eval()
    with torch.no_grad():
        for val_step, (images, box_mask_pairs) in enumerate(tqdm(val_loader)):        
            images = images.to(device)
            
            boxes_batch = []
            masks_batch = []
            box_mask_pairs = box_mask_pairs[0]
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks_item = box_mask_pairs[idx]['masks']
                boxes_xyxy = []
                masks = []
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    box = box[:4]                   
                    boxes_xyxy.append(box)            
                    masks.append(masks_item[i].cuda())
                boxes_xyxy = np.array(boxes_xyxy)   
                H, W = images.shape[-2], images.shape[-1]
                sam_trans = ResizeLongestSide(net.sam.image_encoder.img_size)
                boxes_trans = sam_trans.apply_boxes(boxes_xyxy, (H, W))
                boxes_trans = torch.as_tensor(boxes_trans, dtype=torch.float, device=device)
            
                boxes_batch.append(boxes_trans)
                masks_batch.append(masks)               
                
            predictions = net.forward(images, boxes_batch)
            start_x = int(opt.INPUT_SIZE / 3.0) // 2
            end_x = opt.INPUT_SIZE -1 -start_x
            # masks = masks[:, :, :, start_x:end_x].contiguous()
            # predictions = predictions[:, :, :, start_x:end_x].contiguous()

            # eval metric
            pred_masks_for_vis = torch.zeros(H, W).cuda()                                        
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks = masks_batch[idx]
                pred_masks = predictions[idx]
                #ipdb.set_trace()
                for i in range(len(boxes_item)):
                    pred_masks_for_vis += masks[i]#(torch.sigmoid(pred_masks[i])>0.5).contiguous()
            
            fig, ax = plt.subplots(1)

            pred_masks_for_vis = pred_masks_for_vis.cpu().numpy().astype(np.uint8)
            # Display original image
            ax.imshow(images.squeeze(0).cpu().numpy().transpose((1,2,0)))


            #Display the segmentation mask
            #pred_mask[pred_mask>1] = 1
            show_mask(pred_masks_for_vis, ax)
            #ipdb.set_trace()
            # for i in range(boxes_batch[0].shape[0]):
            # # Display the bounding box
            #     show_box(boxes_batch[0][i].cpu().numpy(), ax)
            #ax.axis("off")
            #ax.set_xlim(0, images.shape[1])
            #ax.set_ylim(images.shape[0], 0)      

            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())      
            # Save the figure
            save_path = os.path.join("./output_vis_CVPR2024/gt", f"result_{val_step}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # if fig.dtype == np.float32 or fig.dtype == np.float64:
            #     fig = np.clip(fig, 0.0, 1.0)

            # # Convert to uint8 if the image is integer type
            # if fig.dtype != np.uint8:
            #     fig = np.clip(fig, 0, 255).astype(np.uint8)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)


def evaluate(net, val_loader, device, opt):
    dice_ = [[], [], [], [], [], [], [], [], [], [], [], []]
    net.eval()
    with torch.no_grad():
        for val_step, (images, box_mask_pairs) in enumerate(tqdm(val_loader)):        
            images = images.to(device)
            
            boxes_batch = []
            masks_batch = []
            box_mask_pairs = box_mask_pairs[0]
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks_item = box_mask_pairs[idx]['masks']
                boxes_xyxy = []
                masks = []
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    box = box[:4]                   
                    boxes_xyxy.append(box)            
                    masks.append(masks_item[i].cuda())
                boxes_xyxy = np.array(boxes_xyxy)   
                H, W = images.shape[-2], images.shape[-1]
                sam_trans = ResizeLongestSide(net.sam.image_encoder.img_size)
                boxes_trans = sam_trans.apply_boxes(boxes_xyxy, (H, W))
                boxes_trans = torch.as_tensor(boxes_trans, dtype=torch.float, device=device)
            
                boxes_batch.append(boxes_trans)
                masks_batch.append(masks)               
                
            predictions = net.forward(images, boxes_batch)
            start_x = int(opt.INPUT_SIZE / 3.0) // 2
            end_x = opt.INPUT_SIZE -1 -start_x
            # masks = masks[:, :, :, start_x:end_x].contiguous()
            # predictions = predictions[:, :, :, start_x:end_x].contiguous()

            # eval metric                                        
            for idx in range(len(box_mask_pairs)):
                boxes_item = box_mask_pairs[idx]['boxes']
                masks = masks_batch[idx]
                pred_masks = predictions[idx]
                for i in range(len(boxes_item)):
                    box = boxes_item[i]
                    label = box[-1]
                    # print(masks[i].shape, pred_masks[i].shape)
                    dice_iter = compute_dice_accuracy(masks[i][:, start_x:end_x].unsqueeze(0).contiguous(), 
                                                      (torch.sigmoid(pred_masks[i])>0.5)[:, start_x:end_x].unsqueeze(0).contiguous())
                    dice_[label].append(dice_iter.cpu().item())             

    # store in dict
    avg_list = []    
    metrics_dict = {}
    for i in range(len(label_list)):
        if len(dice_[i]) == 0:
            d = torch.tensor(0)
        else:
            d = torch.mean(torch.tensor(dice_[i]))
        metrics_dict[label_list[i]] = d
        avg_list.append(d)
    metrics_dict['avg'] = torch.mean(torch.tensor(avg_list))
    metrics_dict['avg(exclude_bg)'] = torch.mean(torch.tensor(avg_list[1:]))

    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sam_box.yaml")
    parser.add_argument("--save_path", type=str, default="./Adapter_relation_20EPOCH")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # opt = Config(config_path=args.config)
    opt = Config_SAM(config_path=args.config)

    # dataset
    test_dataset = SAM_DebrisDataset(root_path=opt.DATA_PATH, image_list=os.path.join(opt.IMAGE_LIST_PATH, 'test.txt'),
                                input_size=opt.INPUT_SIZE, use_augment=False)
    test_loader = DataLoader(test_dataset, batch_size=opt.VAL_BATCHSIZE, shuffle=False, collate_fn=collate_fn_seq_box_seg_pair)

    rand_seed(opt.RANDOM_SEED)

    net = SonarSAM(model_name=opt.SAM_NAME, checkpoint=opt.SAM_CHECKPOINT, num_classes=opt.OUTPUT_CHN, 
                   is_finetune_image_encoder=opt.IS_FINETUNE_IMAGE_ENCODER,
                   use_adaptation=opt.USE_ADAPTATION, 
                   adaptation_type=opt.ADAPTATION_TYPE,
                   head_type=opt.HEAD_TYPE,
                   reduction=4, upsample_times=2, groups=4)
    net = ModelWithLoss(net)

    ckpt = torch.load(os.path.join(args.save_path, '{}_best.pth'.format(opt.MODEL_NAME)))

    net.load_state_dict(ckpt['state_dict'])

    net.to(device)
    #evaluate_vis(net.model, test_loader, device, opt)
    evaluate_image(net.model, test_loader, device, opt)
    evaluate_gt(net.model, test_loader, device, opt)
    # metrics_dict = evaluate(net.model, test_loader, device, opt)
    # print("Dice on Test set:")
    # for key in metrics_dict.keys():
    #     print("{}:\t{:.2f}".format(key, metrics_dict[key]*100))
