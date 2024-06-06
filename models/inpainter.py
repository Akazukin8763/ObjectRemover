import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
propainter_path = os.path.join(current_dir, '../ProPainter')
sys.path.append(propainter_path)

import cv2
import numpy as np
import scipy.ndimage
import torch
from PIL import Image
from tqdm import tqdm

from ProPainter.core.utils import to_tensors
from ProPainter.model.modules.flow_comp_raft import RAFT_bi
from ProPainter.model.propainter import InpaintGenerator
from ProPainter.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from ProPainter.utils.download_util import load_file_from_url

pretrained_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
    
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
parser.add_argument(
    '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
parser.add_argument(
    '-o', '--output', type=str, default='results', help='Output folder. Default: results')
parser.add_argument(
    "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
parser.add_argument(
    '--height', type=int, default=-1, help='Height of the processing video.')
parser.add_argument(
    '--width', type=int, default=-1, help='Width of the processing video.')
parser.add_argument(
    '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
parser.add_argument(
    "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
parser.add_argument(
    "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
parser.add_argument(
    "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
parser.add_argument(
    "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
parser.add_argument(
    '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
parser.add_argument(
    '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
parser.add_argument(
    '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
parser.add_argument(
    '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
parser.add_argument(
    '--save_frames', action='store_true', help='Save output frames. Default: False')
parser.add_argument(
    '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

args = parser.parse_args()


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask


class Inpainter():
    def __init__(self):
        self.model_propainter = None
        self.model_raft = None
        self.model_flow = None

    def load_model(self, gpu: bool = False):
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

        model_names = ['ProPainter', 'raft-things', 'recurrent_flow_completion']
        for model_name in model_names:
            filepath = f'./weights/{model_name}.pth'

            if os.path.isfile(filepath):
                print(f'{filepath} has already existed.')
            else:
                print(f'Download model {filepath}')
                load_file_from_url(url=os.path.join(pretrained_model_url, f'{model_name}.pth'),
                                model_dir='./weights/', progress=True, file_name=None)

        # Load pretrained model
        self.model_propainter = InpaintGenerator(model_path='./weights/ProPainter.pth')
        self.model_raft = RAFT_bi(model_path='./weights/raft-things.pth')
        self.model_flow = RecurrentFlowCompleteNet(model_path='./weights/recurrent_flow_completion.pth')

        # Set to evaluation model
        self.model_propainter = self.model_propainter.to(self.device)
        self.model_propainter.eval()

        self.model_raft = self.model_raft.to(self.device)

        for param in self.model_flow.parameters():
            param.requires_grad = False
        self.model_flow = self.model_flow.to(self.device)
        self.model_flow.eval()

    def predict(self, images, masks):
        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if args.fp16 else False 
        if self.device == torch.device('cpu'):
            use_half = False

        # Convert images to PIL format and resize
        frames = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        size = frames[0].size

        frames, size, out_size = resize_frames(frames, size)
        length = len(frames)
        w, h = size
        
        # Process masks
        masks_img = []
        masks_dilated = []
        flow_masks = []

        for mask in masks:
            mask_img = Image.fromarray(mask)
            mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))

            flow_mask_dilates = 8
            mask_dilates = 5

            # Dilate 8 pixel so that all known pixel is trustworthy
            if flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
            else:
                flow_mask_img = binary_mask(mask_img).astype(np.uint8)
            # Close the small holes inside the foreground objects
            # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
            # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(flow_mask_img * 255))
            
            if mask_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
            else:
                mask_img = binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(mask_img * 255))
        
        if len(masks_img) == 1:
            flow_masks = flow_masks * length
            masks_dilated = masks_dilated * length
        
        # for saving the masked frames or video
        masked_frame_for_save = []
        for i in range(len(frames)):
            mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
            img = np.array(frames[i])
            green = np.zeros([h, w, 3]) 
            green[:,:,1] = 255
            alpha = 0.6
            # alpha = 1.0
            fuse_img = (1-alpha)*img + alpha*green
            fuse_img = mask_ * fuse_img + (1-mask_)*img
            masked_frame_for_save.append(fuse_img.astype(np.uint8))

        frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(self.device), flow_masks.to(self.device), masks_dilated.to(self.device)
        
        ##############################################
        # ProPainter inference
        ##############################################
        video_length = frames.size(1)
        print(f'\nProcessing {video_length} frames...')
        with torch.no_grad():
            # ---- compute flow ----
            if frames.size(-1) <= 640: 
                short_clip_len = 12
            elif frames.size(-1) <= 720: 
                short_clip_len = 8
            elif frames.size(-1) <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2
            
            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.model_raft(frames[:,f:end_f], iters=args.raft_iter)
                    else:
                        flows_f, flows_b = self.model_raft(frames[:,f-1:end_f], iters=args.raft_iter)
                    
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                    
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.model_raft(frames, iters=args.raft_iter)
                torch.cuda.empty_cache()


            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.model_flow = self.model_flow.half()
                self.model_propainter = self.model_propainter.half()

            
            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > args.subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, args.subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + args.subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                    pred_flows_bi_sub, _ = self.model_flow.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        flow_masks[:, s_f:e_f+1])
                    pred_flows_bi_sub = self.model_flow.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        pred_flows_bi_sub, 
                        flow_masks[:, s_f:e_f+1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                    
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = self.model_flow.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.model_flow.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()
                

            # ---- image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                    prop_imgs_sub, updated_local_masks_sub = self.model_propainter.img_propagation(masked_frames[:, s_f:e_f], 
                                                                        pred_flows_bi_sub, 
                                                                        masks_dilated[:, s_f:e_f], 
                                                                        'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                    
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model_propainter.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

        ori_frames = frames_inp
        comp_frames = [None] * video_length

        neighbor_stride = args.neighbor_length // 2
        if video_length > args.subvideo_length:
            ref_num = args.subvideo_length // args.ref_stride
        else:
            ref_num = -1
        
        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = self.model_propainter(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

                pred_img = pred_img.view(-1, 3, h, w)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else: 
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

            torch.cuda.empty_cache()
        
        # save videos frame
        # masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
        comp_frames = [cv2.cvtColor(cv2.resize(f, out_size), cv2.COLOR_RGB2BGR) for f in comp_frames]

        torch.cuda.empty_cache()

        return comp_frames
