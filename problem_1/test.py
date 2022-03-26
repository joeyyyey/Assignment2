import torch
from torch import nn
import numpy as np
import math
from glob import glob
from torch.utils.data import DataLoader
import model

from dataset import LAHeart # added
from dataset import read_h5
from pathlib import Path

from medpy import metric

from model import UNet

# if __name__ == '__main__':

    # use_cuda = False

    # model = UNet() # load your model here

    # patch_size = (112, 112, 80)
    # stride_xy = 18
    # stride_z = 4

    # path_list = glob('./datas/test/*.h5')
    # p = Path(Path(__file__).resolve().parent, 'problem1_datas', 'test')
    # path_list = p.glob('*.h5')

    # print(path_list)
def test(model):
    # testing
    model.eval()

    test_correct = 0
    total = 0
    acc_test = 0
    acc_test_list = []

    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4

    p = Path(Path(__file__).resolve().parent, 'problem1_datas', 'test')
    path_list = p.glob('*.h5')

    for path in path_list:
        images, labels = read_h5(path)
        # if use_cuda:
        #     images = images.cuda()
        #     labels = labels.cuda()

        w, h, d = images.shape
        sx = math.ceil((w - patch_size[0]) / stride_xy) + 1
        sy = math.ceil((h - patch_size[1]) / stride_xy) + 1
        sz = math.ceil((d - patch_size[2]) / stride_z) + 1

        scores = np.zeros((2, ) + images.shape).astype(np.float32)
        counts = np.zeros(images.shape).astype(np.float32)
            
        # inference all windows (patches)
        for x in range(0, sx):
            xs = min(stride_xy * x, w - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, h - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_z * z, d - patch_size[2])

                    # extract one patch for model inference
                    test_patch = images[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    with torch.no_grad():
                        test_patch = torch.from_numpy(test_patch).cuda() # if use cuda
                        test_patch = test_patch.unsqueeze(0).unsqueeze(0) # [1, 1, w, h, d]
                        out = model(test_patch)
                        out = torch.softmax(out, dim=1)
                        out = out.cpu().data.numpy() # [1, 2, w, h, d]
                        
                    # record the predicted scores
                    scores[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out[0, ...]
                    counts[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
            
        scores = scores / np.expand_dims(counts, axis=0)
        predictions = np.argmax(scores, axis = 0) # final prediction: [w, h, d]

        metrics_dc = metric.binary.dc(predictions,labels)
        metrics_jc = metric.binary.jc(predictions,labels)
        # metrics_asd = metric.binary.asd(predictions,labels)
        # metrics_hd95 = metric.binary.hd95(predictions,labels)

    # acc_test = test_correct / len(read_h5(path))
    # acc_test_list.append(acc_test.item())
    print("dc {:.4f}, jc {:.4f}".format(
        metrics_dc,
        metrics_jc,
        # metrics_asd,
        # metrics_hd95,

    ))
