# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
import pdb

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, regress_loss, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    final_losses = AverageMeter()
    reg_losses = AverageMeter()
    mse_losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, cord, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, locs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loc = locs[0]
            loc_x = loc[:, 0:17, :, :]
            loc_y = loc[:, 17:, :, :]
            loc = torch.cat((torch.unsqueeze((loc_x), 0),
                             torch.unsqueeze((loc_y), 0)))
            mse_loss = criterion(outputs[0], target, target_weight)
            mse_reg_loss = regress_loss(loc, cord, target_weight)
            for output, loc in outputs[1:], locs[1:]:
                loc_x = loc[:, 0:17, :, :]
                loc_y = loc[:, 17:, :, :]
                loc = torch.cat((torch.unsqueeze((loc_x), 0),
                                 torch.unsqueeze((loc_y), 0)))
                mse_loss += criterion(output, target, target_weight)
                reg_loss += regress_loss(loc, cord, target_weight)
        else:
            output = outputs
            loc = locs
            loc_x = loc[:, 0:17, :, :]
            loc_y = loc[:, 17:, :, :]
            loc = torch.cat((torch.unsqueeze((loc_x), 0),
                             torch.unsqueeze((loc_y), 0)))
            mse_loss = criterion(output, target, target_weight)
            reg_loss = regress_loss(loc, cord, target_weight)
        final_loss = mse_loss+config.FCOS_WEIGHT*reg_loss

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        final_losses.update(final_loss.item(), input.size(0))
        reg_losses.update(reg_loss.item(), input.size(0))
        mse_losses.update(mse_loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Reg Loss {Rloss.val:.5f} ({Rloss.avg:.5f})\t' \
                  'MSE Loss {MSEloss.val:.5f} ({MSEloss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=final_losses, Rloss=reg_losses, MSEloss=mse_losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', final_losses.val, global_steps)
            writer.add_scalar('train_reg_loss', reg_losses.val, global_steps)
            writer.add_scalar('train_mse_loss', mse_losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, regress_loss, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    final_losses = AverageMeter()
    reg_losses = AverageMeter()
    mse_losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, cord, meta) in enumerate(val_loader):
            # compute output
            outputs, locs = model(input)
            loc_x = locs[:, 0:17, :, :]
            loc_y = locs[:, 17:, :, :]
            locs = torch.cat((torch.unsqueeze((loc_x), 0),
                              torch.unsqueeze((loc_y), 0)))
            xlocs = torch.squeeze((locs[0]), 0)
            ylocs = torch.squeeze((locs[1]), 0)

            if isinstance(outputs, list):
                output = outputs[-1]
                xloc = xlocs[-1]
                yloc = ylocs[-1]
            else:
                output = outputs
                xloc = xlocs
                yloc = ylocs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped, locs_flipped = model(input_flipped)
                xlocs_flipped = torch.squeeze((locs[0]), 0)
                ylocs_flipped = torch.squeeze((locs[1]), 0)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                    xloc_flipped = xlocs_flipped[-1]
                    yloc_flipped = xlocs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                    xloc_flipped = xlocs_flipped
                    yloc_flipped = ylocs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5
                xloc = (xloc+xloc_flipped)*0.5
                yloc = (yloc+yloc_flipped)*0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loc = torch.cat((torch.unsqueeze((xloc), 0),
                             torch.unsqueeze((yloc), 0)))

            mse_loss = criterion(output, target, target_weight)
            reg_loss = regress_loss(loc, cord, target_weight)
            final_loss = mse_loss+0.01*reg_loss

            num_images = input.size(0)
            # measure accuracy and record loss
            final_losses.update(final_loss.item(), input.size(0))
            reg_losses.update(reg_loss.item(), input.size(0))
            mse_losses.update(mse_loss.item(), input.size(0))
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(),
                xloc.clone().cpu().numpy(),
                yloc.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'MSE Loss {mse_loss.val:.4f} ({mse_loss.avg:.4f})\t' \
                      'Reg Loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=final_losses, mse_loss=mse_losses, reg_loss=reg_losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_final_loss',
                final_losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_mse_loss',
                mse_losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_reg_loss',
                reg_losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator


def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
