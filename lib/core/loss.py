# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class RegLoss(nn.Module):
    # TODO： Set the weight decay related to the distance to the joint
    def __init__(self, use_target_weight):
        super(RegLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, loc, cord, target_weight):
        batch_size = loc.size(1)
        num_joints = loc.size(2)
        y_axis = loc.size(3)
        x_axis = loc.size(4)
        loc_x = loc[0]
        loc_y = loc[1]
        loc_x = loc_x.reshape((batch_size, num_joints, -1)).split(1, 1)
        loc_y = loc_y.reshape((batch_size, num_joints, -1)).split(1, 1)
        joint = cord.reshape((2, num_joints, batch_size)).split(1, 1)
        gt_shift_x = np.zeros(
            (num_joints, batch_size, y_axis, x_axis), dtype=np.float32)
        gt_shift_y = np.zeros(
            (num_joints, batch_size, y_axis, x_axis), dtype=np.float32)
        gt_shift_x = torch.from_numpy(gt_shift_x)
        gt_shift_y = torch.from_numpy(gt_shift_y)
        loss = 0

        for idx in range(num_joints):
            idx_joint = joint[idx]
            idx_joint = torch.squeeze(idx_joint, 1)
            for xi in range(x_axis):
                idx_batch = xi-idx_joint[0]
                shift_x_value = np.zeros((batch_size, y_axis), dtype=np.int32)
                for bi in range(batch_size):
                    shift_bi_value = idx_batch[bi]
                    shift_bi = np.full(
                        (y_axis), shift_bi_value, dtype=np.int32)
                    shift_x_value[bi] = shift_bi
                shift_x_value = torch.from_numpy(shift_x_value)
                shift_x_value = torch.unsqueeze(shift_x_value, 2)
                if xi == 0:
                    gt_shift_idx = shift_x_value
                else:
                    gt_shift_idx = torch.cat((gt_shift_idx, shift_x_value), 2)
            gt_shift_x[idx] = gt_shift_idx
            for yi in range(y_axis):
                idx_batch = yi-idx_joint[1]
                shift_y_value = np.zeros((batch_size, x_axis), dtype=np.int32)
                for bi in range(batch_size):
                    shift_bi_value = idx_batch[bi]
                    shift_bi = np.full(
                        (x_axis), shift_bi_value, dtype=np.int32)
                    shift_y_value[bi] = shift_bi
                shift_y_value = torch.from_numpy(shift_y_value)
                shift_y_value = torch.unsqueeze(shift_y_value, 1)
                if yi == 0:
                    gt_shift_idx = shift_y_value
                else:
                    gt_shift_idx = torch.cat((gt_shift_idx, shift_y_value), 1)
            gt_shift_y[idx] = gt_shift_idx

        gt_shift_x = gt_shift_x.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        gt_shift_y = gt_shift_y.reshape(
            (batch_size, num_joints, -1)).split(1, 1)

        for idx in range(num_joints):
            loc_x_pred = loc_x[idx].squeeze()

            loc_y_pred = loc_y[idx].squeeze()
            gt_shift_xi = gt_shift_x[idx].squeeze().cuda()
            gt_shift_yi = gt_shift_y[idx].squeeze().cuda()
            aa = loc_x_pred.mul(target_weight[:, idx])
            bb = gt_shift_xi.mul(target_weight[:, idx])
            if self.use_target_weight:
                loss += 0.5*(self.criterion(
                    loc_x_pred.mul(target_weight[:, idx]),
                    gt_shift_xi.mul(target_weight[:, idx])) +
                    self.criterion(
                    loc_y_pred.mul(target_weight[:, idx]),
                    gt_shift_yi.mul(target_weight[:, idx]))
                )
            else:
                loss += 0.5*(self.criterion(loc_x_pred, gt_shift_xi) +
                             sel.criterion(loc_y_pred, gt_shift_yi))
        return loss/num_joints


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
