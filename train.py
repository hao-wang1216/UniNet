import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '12355'

import torch
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='nccl', world_size=1, rank=0)
import torch
import torch.nn.functional as F
from datetime import datetime
import logging
import argparse
import os
import numpy as np

import UniNet


from dataset_train import get_loader

from torch.utils.tensorboard import SummaryWriter


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        # Full checkpoint with model, optimizer, and training state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint.get('epoch', 1)  # Default to epoch 1 if not found
            loss = checkpoint.get('loss', 0)    # Default to 0 if not found
            return epoch, loss
        else:
            # Might be a state_dict directly in the dict
            model.load_state_dict(checkpoint)
            return 1, 0
    else:
        # Assume it's just the model state_dict
        model.load_state_dict(checkpoint)
        return 1, 0

def train(train_loader, model, optim, epoch, opt, total_step, writer):
    total_loss = 0
    model.train()
    for step, data in enumerate(train_loader):
        imgs, gts, pianzhen = data
        imgs = imgs.cuda()
        gts = gts.cuda()
        pianzhen = pianzhen.cuda()
        
        input_fea = torch.cat((imgs, pianzhen), dim=0)

        optim.zero_grad()
        stage_pre, pre = model(input_fea)
        stage_loss_list = [structure_loss(out, gts) for out in stage_pre]
        stage_loss = 0
        gamma = 0.2
        for iteration in range(len(stage_pre)):
            stage_loss += (gamma * iteration) * stage_loss_list[iteration]

        map_loss = structure_loss(pre, gts)
        loss = stage_loss + map_loss
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optim.step()

        total_loss += loss

        if step % 1000 == 0 or step == total_step:
            print(
                '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss: {:0.4f}]'.
                format(datetime.now(), epoch, opt.epoch, step, total_step, loss.item()))
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:0.4f}'.
                format(epoch, opt.epoch, step, total_step, loss.item()))

    writer.add_scalar("Train_Loss", total_loss, global_step=epoch)

    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        checkpoint = {
            'epoch': epoch + 1,  # 保存下一个epoch作为起始点
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        # 保存检查点，文件名包含epoch信息
        #torch.save(checkpoint, save_path + f'UniNet_epoch_{epoch}.pth')
        print(f"Checkpoint saved: {save_path}UniNet_epoch_{epoch}.pth")

def val(val_loader, model, epoch, opt, writer):
    model.eval()
    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for step, data in enumerate(val_loader):
            imgs, gts, pianzhen = data
            imgs = imgs.cuda()
            gts = gts.cuda()
            pianzhen = pianzhen.cuda()
            input_fea = torch.cat((imgs, pianzhen), dim=0)

            _, pre = model(input_fea)
            pre = torch.sigmoid(pre)

            # Calculate MAE (Mean Absolute Error)
            mae = torch.abs(pre - gts).mean()
            total_mae += mae.item() * imgs.size(0)
            total_samples += imgs.size(0)

    avg_mae = total_mae / total_samples
    print(
        '[{}] => [Epoch Num: {:03d}/{:03d}] => [Val MAE: {:0.4f}]'.
        format(datetime.now(), epoch, opt.epoch, avg_mae))
    logging.info(
        '#VAL#:Epoch [{:03d}/{:03d}], MAE: {:0.4f}'.
        format(epoch, opt.epoch, avg_mae))

    writer.add_scalar("Val_MAE", avg_mae, global_step=epoch)
    return avg_mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='choosing optimizer Adam')
    parser.add_argument('--augmentation', default=False, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=704, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str, default='/root/Camoflaged/data/RGB-D/COD-TrainDataset', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='/root/Camoflaged/data/RGB-D/COD-TestDataset/CAMO', help='path to validation dataset')
    
    # parser.add_argument('--train_path', type=str, default='/root/Camoflaged/data/RGB-P/train/', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='/root/Camoflaged/data/RGB-P/test/', help='path to validation dataset')

    # parser.add_argument('--train_path', type=str, default='/root/Camoflaged/data/RGB-P/train/', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='/root/Camoflaged/data/RGB-P/test/', help='path to validation dataset')
    
    # parser.add_argument('--train_path', type=str, default='/media/user/d/hao/tem/VT5000/Train/', help='path to train dataset')
    # parser.add_argument('--val_path', type=str, default='/media/user/d/hao/tem/VT5000/Test/', help='path to validation dataset')
    parser.add_argument('--save_path', type=str, default='/root/Camoflaged/UniNet/pth', help='path to save your model')
    parser.add_argument('--epoch_save', type=int, default=20, help='every n epochs to save model')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume training')
    opt = parser.parse_args()

    # 确保保存路径以斜杠结尾
    if not opt.save_path.endswith('/'):
        opt.save_path += '/'
    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(filename=opt.save_path+'Plog.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("CCD-Train")

    model = UniNet().cuda()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 1
    best_mae = float('inf')

    # 加载检查点（如果提供）
    if opt.resume:
        print(f"Attempting to load checkpoint from {opt.resume}")
        try:
            start_epoch, _ = load_checkpoint(model, optimizer, opt.resume)
            print(f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}")
            
            # 如果从检查点恢复，尝试加载之前的最佳MAE
            checkpoint_dir = os.path.dirname(opt.resume)
            best_model_path = os.path.join(checkpoint_dir, 'DformerUniNet.pth')
            if os.path.exists(best_model_path):
                print(f"Found previous best model at {best_model_path}")
                # 这里可以添加代码来验证或加载之前的最佳MAE值
                
        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            print("Starting training from scratch")
            start_epoch = 1
            
    # train_image_root = '{}/Imgs/'.format(opt.train_path)
    # train_gt_root = '{}/GT/'.format(opt.train_path)
    # train_pianzhen_root = '{}/depth/'.format(opt.train_path)

    train_image_root = '{}/train-rgb/'.format(opt.train_path)
    train_gt_root = '{}/train-gt/'.format(opt.train_path)
    train_pianzhen_root = '{}/train-dop/'.format(opt.train_path)

    # train_image_root = '{}/RGB/'.format(opt.train_path)
    # train_gt_root = '{}/GT/'.format(opt.train_path)
    # train_pianzhen_root = '{}/T/'.format(opt.train_path)
    train_loader = get_loader(train_image_root, train_gt_root, train_pianzhen_root, batch_size=opt.batchsize, image_size=opt.trainsize, num_workers=2)
    total_step = len(train_loader)

    # val_image_root = '{}/Imgs/'.format(opt.val_path)
    # val_gt_root = '{}/GT/'.format(opt.val_path)
    # val_pianzhen_root = '{}/depth/'.format(opt.val_path)
    
    val_image_root = '{}/test-rgb/'.format(opt.val_path)
    val_gt_root = '{}/test-gt/'.format(opt.val_path)
    val_pianzhen_root = '{}/test-dop/'.format(opt.val_path)

    # val_image_root = '{}/RGB/'.format(opt.val_path)
    # val_gt_root = '{}/GT/'.format(opt.val_path)
    # val_pianzhen_root = '{}/T/'.format(opt.val_path)
    val_loader = get_loader(val_image_root, val_gt_root, val_pianzhen_root, batch_size=opt.batchsize, image_size=opt.trainsize, num_workers=2)

    writer = SummaryWriter(opt.save_path + "SummaryWriter")

    print('--------------------training----------------------')
    print(f'Starting from epoch: {start_epoch}')

    for epoch in range(start_epoch, opt.epoch+1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, opt, total_step, writer)
        current_mae = val(val_loader, model, epoch, opt, writer)

        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), opt.save_path + 'D_UniNet.pth')
            print(f"New best model saved with MAE: {best_mae:.4f}")
            
            # 同时保存完整的检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_mae,
                'best_mae': best_mae
            }
            torch.save(checkpoint, opt.save_path + f'P_best_epoch_{epoch}.pth')
    
    writer.close()