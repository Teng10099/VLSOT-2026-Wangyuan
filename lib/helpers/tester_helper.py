import os
import tqdm
import json

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.utils_helper import calc_iou
import time
import numpy as np
from utils import misc
import torch.nn.functional as F

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, loss, train_cfg=None, model_name='mono3dvg'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs  # max objects per image, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('.\\' + train_cfg['save_path'], model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.model_name = model_name
        self.mono3dvg_loss = loss
        
        # === 新增部分：初始化单目标跟踪指标 ===
        self.total_iou = 0.0  # 累积IoU
        self.total_center_error = 0.0  # 累积中心误差
        self.total_frames = 0  # 总帧数
        self.success_sr50 = 0  # 成功率 (IoU > 0.5)
        self.success_sr60 = 0  # 成功率 (IoU > 0.25)
        self.success_sr70 = 0  # 成功率 (IoU > 0.5)
        self.success_sr80 = 0  # 成功率 (IoU > 0.25)
        self.success_sr90 = 0  # 成功率 (IoU > 0.5)
        self.precision_pr20 = 0  # 精确率 (中心误差 < 1米)
        self.precision_pr10 = 0  # 精确率 (中心误差 < 1米)
        self.precision_pr05 = 0  # 精确率 (中心误差 < 0.5米)
        # === 新增结束 ===

    def test(self):
        # Test a checkpoint
        # checkpoint_path = os.path.join(self.cfg['pretrain_model'])
        # assert os.path.exists(checkpoint_path)
        # load_checkpoint(model=self.model,
        #                 optimizer=None,
        #                 filename=checkpoint_path,
        #                 map_location=self.device,
        #                 logger=self.logger)
        self.model.to(self.device)
        self.inference()

    def inference(self):
        torch.set_grad_enabled(False)
        save_root = "./track_txt"
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)
        self.model.eval()

        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0

        # Initialize accumulators for losses
        avg_loss_ce = avg_loss_bbox = avg_loss_giou = avg_loss_depth = 0
        avg_loss_dim = avg_loss_angle = avg_loss_center = avg_loss_depth_map = avg_loss_mono3dvg = 0

        count = 0  # To track frames processed for averaging losses

        for batch_idx, (inputs, calibs, targets,videoID) in enumerate(self.dataloader):
            captions = targets["text"]["text"]
            for key in targets.keys():
                if key not in ['text']:
                    other_dict = targets[key]
                    for j in range(len(other_dict)):
                        single_dict = other_dict[j] 
                        for key in single_dict.keys():
                            single_dict[key] = single_dict[key].to(self.device)
            inputs = torch.stack(inputs, 1)#torch.Size([1, 3, 384, 1280])
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = targets["other"][0]['img_size'].to(self.device)
            targets = self.prepare_targets(targets, inputs.shape[0])
            lines = []
            start_time = time.time()

            for t in range(inputs.size(1)):  # Iterate over 30 frames
                tinput = F.interpolate(inputs[:, t, :, :, :], scale_factor=1, mode='bilinear', align_corners=False)
                outputs = self.model(tinput, calibs, img_sizes, captions)

                # Compute losses
                mono3dvg_losses_dict = self.mono3dvg_loss(outputs, targets[t])
                weight_dict = self.mono3dvg_loss.weight_dict
                mono3dvg_losses_dict_weighted = [mono3dvg_losses_dict[k] * weight_dict[k] for k in mono3dvg_losses_dict.keys() if k in weight_dict]
                mono3dvg_losses = sum(mono3dvg_losses_dict_weighted)

                mono3dvg_losses_dict = misc.reduce_dict(mono3dvg_losses_dict)
                mono3dvg_losses_dict_log = {}
                mono3dvg_losses_log = 0
                for k in mono3dvg_losses_dict.keys():
                    if k in weight_dict:
                        mono3dvg_losses_dict_log[k] = (mono3dvg_losses_dict[k] * weight_dict[k]).item()
                        mono3dvg_losses_log += mono3dvg_losses_dict_log[k]
                mono3dvg_losses_dict_log["loss_mono3dvg"] = mono3dvg_losses_log

                # Accumulate losses for averaging across 30 frames
                avg_loss_ce += mono3dvg_losses_dict_log['loss_ce']
                avg_loss_bbox += mono3dvg_losses_dict_log['loss_bbox']
                avg_loss_giou += mono3dvg_losses_dict_log['loss_giou']
                avg_loss_depth += mono3dvg_losses_dict_log['loss_depth']
                avg_loss_dim += mono3dvg_losses_dict_log['loss_dim']
                avg_loss_angle += mono3dvg_losses_dict_log['loss_angle']
                avg_loss_center += mono3dvg_losses_dict_log['loss_center']
                avg_loss_depth_map += mono3dvg_losses_dict_log['loss_depth_map']
                avg_loss_mono3dvg += mono3dvg_losses_dict_log["loss_mono3dvg"]
                
                # === 新增部分：计算单目标跟踪指标 === 

                # 获取预测的3D包围框坐标（[x_min, y_min, z_min, x_max, y_max, z_max]）
                pred_boxes_3d = outputs['pred_boxes']  # shape [1, N, 6]
                pred_boxes_3d = pred_boxes_3d.squeeze(1)  # shape [N, 6]

                # 获取真实的3D包围框坐标（[x_min, y_min, z_min, x_max, y_max, z_max]）
                gt_boxes_3d = targets[t][0]['boxes_3d']  # shape [6]

                if pred_boxes_3d.size(0) > 0 and gt_boxes_3d.size(0) > 0:
                    # 计算预测框和真实框的中心坐标和尺寸（宽、长、高）
                    
                    # 预测框
                    pred_boxes_min = pred_boxes_3d[:, :3]  # x_min, y_min, z_min
                    pred_boxes_max = pred_boxes_3d[:, 3:]  # x_max, y_max, z_max
                    pred_centers = (pred_boxes_min + pred_boxes_max) / 2  # 中心坐标，shape [N, 3]
                    pred_dims = outputs['pred_3d_dim']  # 尺寸，shape [N, 3]
                    pred_dims = pred_dims.squeeze(1)

                    # 真实框
                    gt_boxes_min = gt_boxes_3d[:, :3]  # x_min, y_min, z_min
                    gt_boxes_max = gt_boxes_3d[:, 3:]  # x_max, y_max, z_max
                    gt_centers = (gt_boxes_min + gt_boxes_max) / 2  # 中心坐标，shape [1, 3]
                    gt_dims = targets[t][0]['size_3d']  # 尺寸，shape [1, 3]
                    gt_dims = gt_dims.squeeze(1)
                    

                    # 准备输入给 calc_iou 的包围框，格式为 [中心坐标, 尺寸]
                    pred_box = torch.cat([pred_centers[0], pred_dims[0]], dim=0)  # shape [6]
                    gt_box = torch.cat([gt_centers[0], gt_dims[0]], dim=0)  # shape [6]

                    # 计算 IoU
                    iou = calc_iou(pred_boxes_3d, gt_boxes_3d)  # 返回标量

                    # 计算中心点误差
                    center_error = torch.norm(pred_centers[0] - gt_centers[0], p=2).item()  # 返回标量

                    # 累积 IoU 和中心误差
                    self.total_iou += iou
                    self.total_center_error += center_error
                    self.total_frames += 1

                    # 成功率（IoU > 0.5）
                    self.success_sr50 += int(iou > 0.50)
                    self.success_sr60 += int(iou > 0.60)
                    self.success_sr70 += int(iou > 0.70)
                    self.success_sr80 += int(iou > 0.80)
                    self.success_sr90 += int(iou > 0.90)

                    # 精确率（中心误差 < 1米）
                    self.precision_pr20 += int(center_error < 2.0)
                    self.precision_pr10 += int(center_error < 1.0)
                    self.precision_pr05 += int(center_error < 0.5)
                    center = pred_centers[0]     # (x, y, z)
                    dims   = pred_dims[0]
                    x, y, z = center.tolist()
                    h,w,l = dims.tolist() 
                    frame_id = t + 1
                    line = f"{frame_id} {abs(l)} {w} {h} {x} {y} {z}"
                    lines.append(line)
                else:
                    # 处理没有检测到目标的情况
                    self.total_frames += 1
                    frame_id = t + 1
                    line = f"{frame_id} 0 0 0 0 0 0"
                    lines.append(line)

                # === 新增结束 ===


                
                count += 1
            txt_path = os.path.join(save_root,videoID[0]+'.txt')
            with open(txt_path,'w') as f:
                f.write("\n".join(lines))
            progress_bar.update()
            avg_loss_ce /= 30
            avg_loss_bbox /= 30
            avg_loss_giou /= 30
            avg_loss_depth /= 30
            avg_loss_dim /= 30
            avg_loss_angle /= 30
            avg_loss_center /= 30
            avg_loss_depth_map /= 30
            avg_loss_mono3dvg /= 30

            # Log the average losses
            print_str = (f'Evaluation: [{batch_idx}/{len(self.dataloader)}]\t'
                        f'Avg_Loss_mono3dvg: {avg_loss_mono3dvg:.2f}\t'
                        f'avg_loss_ce: {avg_loss_ce:.2f}\t'
                        f'avg_loss_bbox: {avg_loss_bbox:.2f}\t'
                        f'avg_loss_giou: {avg_loss_giou:.2f}\t' 
                        f'avg_loss_depth: {avg_loss_depth:.2f}\t'
                        f'avg_loss_dim: {avg_loss_dim:.2f}\t'
                        f'avg_loss_angle: {avg_loss_angle:.2f}\t'
                        f'avg_loss_center: {avg_loss_center:.2f}\t'
                        f'avg_loss_depth_map: {avg_loss_depth_map:.2f}')
            self.logger.info(print_str)

            # Reset the average loss accumulators
            avg_loss_ce = avg_loss_bbox = avg_loss_giou = avg_loss_depth = 0
            avg_loss_dim = avg_loss_angle = avg_loss_center = avg_loss_depth_map = avg_loss_mono3dvg = 0
            count = 0

            end_time = time.time()
            model_infer_time += end_time - start_time

            # === 新增部分：计算并记录单目标3D跟踪指标 ===
            if self.total_frames > 0:
                # 平均重叠率 (AOR)
                aor = self.total_iou / self.total_frames

                # 平均中心误差 (ACE)
                ace = self.total_center_error / self.total_frames

                # 成功率 (SR)
                sr50 = (self.success_sr50 / self.total_frames) * 100  # 转换为百分比
                sr60 = (self.success_sr60 / self.total_frames) * 100  # 转换为百分比
                sr70 = (self.success_sr70 / self.total_frames) * 100  # 转换为百分比
                sr80 = (self.success_sr80 / self.total_frames) * 100  # 转换为百分比
                sr90 = (self.success_sr90 / self.total_frames) * 100  # 转换为百分比
                # 精确率 (PR)
                pr20 = (self.precision_pr20 / self.total_frames) * 100  # 转换为百分比
                pr10 = (self.precision_pr10 / self.total_frames) * 100  # 转换为百分比
                pr05 = (self.precision_pr05 / self.total_frames) * 100  # 转换为百分比

                # 记录指标到日志
                self.logger.info(f'AOR (Average Overlap Rate): {aor:.4f}')
                self.logger.info(f'ACE (Average Center Error): {ace:.4f} pixel')
                self.logger.info(f'SR (Success Rate @ IoU>0.5): {sr50:.2f}%')
                self.logger.info(f'SR (Success Rate @ IoU>0.6): {sr60:.2f}%')
                self.logger.info(f'SR (Success Rate @ IoU>0.7): {sr70:.2f}%')
                self.logger.info(f'SR (Success Rate @ IoU>0.8): {sr80:.2f}%')
                self.logger.info(f'SR (Success Rate @ IoU>0.9): {sr90:.2f}%')
                self.logger.info(f'PR (Precision Rate @ ACE<2.0): {pr20:.2f}%')
                self.logger.info(f'PR (Precision Rate @ ACE<1.0): {pr10:.2f}%')
                self.logger.info(f'PR (Precision Rate @ ACE<0.5): {pr05:.2f}%')

            else:
                self.logger.info("No frames were processed for tracking metrics.")

            # === 新增结束 ===

        print(f"Inference on {len(self.dataloader)} objects, average inference time: {model_infer_time / len(self.dataloader)} seconds")
        progress_bar.close()

        # Summarize evaluation results
        self.logger.info('==> Mono3DVG Evaluation Finished')
        return None

    def evaluate(self, epoch):
        torch.set_grad_enabled(False)
        self.model.eval()
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0

        # Initialize accumulators for losses
        avg_loss_ce = avg_loss_bbox = avg_loss_giou = avg_loss_depth = 0
        avg_loss_dim = avg_loss_angle = avg_loss_center = avg_loss_depth_map = avg_loss_mono3dvg = 0

        count = 0  # To track frames processed for averaging losses

        # === 新增部分：初始化单目标跟踪指标 ===
        total_iou = 0.0  # 累积IoU
        total_center_error = 0.0  # 累积中心误差
        total_frames = 0  # 总帧数
        success_sr50 = 0
        success_sr25 = 0  # 成功率 (IoU > 0.5)
        precision_pr10 = 0  # 精确率 (中心误差 < 1米)
        precision_pr05 = 0
        # === 新增结束 ===
        
        for batch_idx, (inputs, calibs, targets) in enumerate(self.dataloader):
            captions = targets["text"]["text"]
            for key in targets.keys():
                if key not in ['text']:
                    other_dict = targets[key]
                    for j in range(len(other_dict)):
                        single_dict = other_dict[j] 
                        for key in single_dict.keys():
                            single_dict[key] = single_dict[key].to(self.device)
            inputs = torch.stack(inputs, 1)#torch.Size([1, 3, 384, 1280])
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = targets["other"][0]['img_size'].to(self.device)
            targets = self.prepare_targets(targets, inputs.shape[0])

            for t in range(inputs.size(1)):  # Iterate over 30 frames
                tinput = F.interpolate(inputs[:, t, :, :, :], scale_factor=1, mode='bilinear', align_corners=False)
                outputs = self.model(tinput, calibs, img_sizes, captions)

                # Compute losses
                mono3dvg_losses_dict = self.mono3dvg_loss(outputs, targets[t])
                weight_dict = self.mono3dvg_loss.weight_dict
                mono3dvg_losses_dict_weighted = [mono3dvg_losses_dict[k] * weight_dict[k] for k in mono3dvg_losses_dict.keys() if k in weight_dict]
                mono3dvg_losses = sum(mono3dvg_losses_dict_weighted)

                mono3dvg_losses_dict = misc.reduce_dict(mono3dvg_losses_dict)
                mono3dvg_losses_dict_log = {}
                mono3dvg_losses_log = 0
                for k in mono3dvg_losses_dict.keys():
                    if k in weight_dict:
                        mono3dvg_losses_dict_log[k] = (mono3dvg_losses_dict[k] * weight_dict[k]).item()
                        mono3dvg_losses_log += mono3dvg_losses_dict_log[k]
                mono3dvg_losses_dict_log["loss_mono3dvg"] = mono3dvg_losses_log

                # === 新增部分：计算单目标跟踪指标 === 

                # 获取预测的3D包围框坐标（[x_min, y_min, z_min, x_max, y_max, z_max]）
                pred_boxes_3d = outputs['pred_boxes']  # shape [1, N, 6]
                pred_boxes_3d = pred_boxes_3d.squeeze(1)  # shape [N, 6]

                # 获取真实的3D包围框坐标（[x_min, y_min, z_min, x_max, y_max, z_max]）
                gt_boxes_3d = targets[t][0]['boxes_3d']  # shape [6]

                if pred_boxes_3d.size(0) > 0 and gt_boxes_3d.size(0) > 0:
                    # 计算预测框和真实框的中心坐标和尺寸（宽、长、高）
                    
                    # 预测框
                    pred_boxes_min = pred_boxes_3d[:, :3]  # x_min, y_min, z_min
                    pred_boxes_max = pred_boxes_3d[:, 3:]  # x_max, y_max, z_max
                    pred_centers = (pred_boxes_min + pred_boxes_max) / 2  # 中心坐标，shape [N, 3]
                    pred_dims = outputs['pred_3d_dim']  # 尺寸，shape [N, 3]
                    pred_dims = pred_dims.squeeze(1)

                    # 真实框
                    gt_boxes_min = gt_boxes_3d[:, :3]  # x_min, y_min, z_min
                    gt_boxes_max = gt_boxes_3d[:, 3:]  # x_max, y_max, z_max
                    gt_centers = (gt_boxes_min + gt_boxes_max) / 2  # 中心坐标，shape [1, 3]
                    gt_dims = targets[t][0]['size_3d']  # 尺寸，shape [1, 3]
                    gt_dims = gt_dims.squeeze(1)
                    

                    # 准备输入给 calc_iou 的包围框，格式为 [中心坐标, 尺寸]
                    pred_box = torch.cat([pred_centers[0], pred_dims[0]], dim=0)  # shape [6]
                    gt_box = torch.cat([gt_centers[0], gt_dims[0]], dim=0)  # shape [6]

                    # 计算 IoU
                    iou = calc_iou(pred_boxes_3d, gt_boxes_3d)  # 返回标量

                    # 计算中心点误差
                    center_error = torch.norm(pred_centers[0] - gt_centers[0], p=2).item()  # 返回标量

                    # 累积 IoU 和中心误差
                    total_iou += iou
                    total_center_error += center_error
                    total_frames += 1

                    # 成功率（IoU > 0.5）
                    success_sr50 += int(iou > 0.5)
                    success_sr25 += int(iou > 0.25)

                    # 精确率（中心误差 < 1米）
                    precision_pr10 += int(center_error < 1.0)
                    precision_pr05 += int(center_error < 0.5)

                else:
                    # 处理没有检测到目标的情况
                    total_frames += 1

                progress_bar.update()
                
            # === 新增部分：计算并记录单目标3D跟踪指标 ===
            if total_frames > 0:
                # 平均重叠率 (AOR)
                aor = total_iou / total_frames

                # 平均中心误差 (ACE)
                ace = total_center_error / total_frames

                # 成功率 (SR)
                sr50 = (success_sr50 / total_frames) * 100  # 转换为百分比
                sr25 = (success_sr25 / total_frames) * 100  # 转换为百分比

                # 精确率 (PR)
                pr10 = (precision_pr10 / total_frames) * 100  # 转换为百分比
                pr05 = (precision_pr05 / total_frames) * 100  # 转换为百分比

                # 记录指标到日志
                self.logger.info(f'AOR (Average Overlap Rate): {aor:.4f}')
                self.logger.info(f'ACE (Average Center Error): {ace:.4f} meters')
                self.logger.info(f'SR (Success Rate @ IoU>0.5): {sr50:.2f}%')
                self.logger.info(f'SR (Success Rate @ IoU>0.25): {sr25:.2f}%')
                self.logger.info(f'PR (Precision Rate @ ACE<1m): {pr10:.2f}%')
                self.logger.info(f'PR (Precision Rate @ ACE<0.5m): {pr05:.2f}%')

            else:
                self.logger.info("No frames were processed for tracking metrics.")

            # === 新增结束 ===

        progress_bar.close()
        return sr25, pr10, aor, ace

    def prepare_targets(self, targets, batch_size):   
        big_targets_list = []
        targets= targets['other']
        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for _ in range(30):
            targets_list = []
            tar = targets[_]
            mask = tar['mask_2d']
            for bz in range(batch_size):
                target_dict = {}
                for key, val in tar.items():
                    if key in key_list:
                        target_dict[key] = val[bz][mask[bz]]
                targets_list.append(target_dict)
            big_targets_list.append(targets_list)
        return big_targets_list        