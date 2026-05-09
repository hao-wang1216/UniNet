import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from Net import UniNet
from dataset_train import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size default 704')
parser.add_argument('--pth_path', type=str, default=r'/root/Camoflaged/UniNet/pth/D_best_epoch_84.pth', help='path to load your model checkpoint')
parser.add_argument('--test_path', type=str, default=r'/root/Camoflaged/data/RGB-D/COD-TestDataset', help='path to test dataset')
# parser.add_argument('--test_path', type=str, default=r'/root/Camoflaged/data/RGB-P/', help='path to test dataset')
opt = parser.parse_args()
#'CAMO','CHAMELEON','COD10K','NC4K'
#test
#VT5000,VT1000,VT821
for _data_name in ['CAMO','CHAMELEON','COD10K','NC4K']:
    data_path = os.path.join(opt.test_path, _data_name)
    save_path = os.path.join(r'/root/Camoflaged/UniNet/Dep_result', _data_name)
    model = UniNet()

    # 加载 checkpoint 文件
    checkpoint = torch.load(opt.pth_path)

    # 判断 checkpoint 是否包含额外的键
    if 'model_state_dict' in checkpoint:
        # 如果是完整的 checkpoint，提取模型权重
        model_state_dict = checkpoint['model_state_dict']
    else:
        # 如果是单独的模型权重，直接使用
        model_state_dict = checkpoint

    # 加载模型权重
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    print("Save path:", save_path)
    
    image_root = os.path.join(data_path, 'Imgs/')
    depth_root = os.path.join(data_path, 'depth/')
    gt_root = os.path.join(data_path, 'GT/')
    
    # image_root = os.path.join(data_path, 'test-rgb/')
    # depth_root = os.path.join(data_path, 'test-dop/')
    # gt_root = os.path.join(data_path, 'test-gt/')

    # image_root = os.path.join(data_path, 'RGB/')
    # depth_root = os.path.join(data_path, 'T/')
    # gt_root = os.path.join(data_path, 'GT/')
    print('root', image_root, depth_root, gt_root)
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    print('****', test_loader.size)
    for i in range(test_loader.size):
        image, gt, depth, name = test_loader.load_data()
        print('***name', name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()

        x_in = torch.cat((image, depth), dim=0)

        P1, P2 = model(x_in)
        res = F.upsample(P1[-1] + P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)

        # 提取文件名并确保以 .png 结尾
        name = os.path.basename(name)
        if not name.endswith('.png'):
            name += '.png'

        # 拼接保存路径
        save_image_path = os.path.join(save_path, name)

        # 保存图片
        success = cv2.imwrite(save_image_path, res)
        if not success:
            print(f"Failed to save image: {save_image_path}")
        else:
            print(f"Saved image: {save_image_path}")