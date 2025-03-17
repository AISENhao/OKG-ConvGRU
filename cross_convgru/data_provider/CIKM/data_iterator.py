import sys
import os
import imageio
from imageio import imsave,imread
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)
# from core.utils.util import *
from torch.utils import data
# from scipy.misc import imsave,imread
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
def sample5(batch_size,mode = 'random',data_type='train',index = None):
    # yao=str(canshu)
    save_root = r'D:\PyCharm projects/cross_convgru/data3/SST/' + data_type + '/'
    # print(save_root)

    if data_type == 'train':
        if mode == 'random':#mode=random；；；；；；；
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,180)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []

                for t in range(1,7):
                    img_path = img_fold + 'img_'+str(t)+'.tif'
                    img = imread(img_path)[:,:,np.newaxis]

                    img = img[0:200, 0:200, :]
                    img = np.clip(img, 0, 255)

                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1#7
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]

                    img = img[0:200, 0:200, :]
                    img = np.clip(img, 0, 255)
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]
                img = img[0:200, 0:200, :]

                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)


    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]
                img = img[0:200, 0:200, :]

                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")
def sample4(batch_size,mode = 'random',data_type='train',index = None):
    # yao=str(canshu)
    save_root = r'D:\PyCharm projects/cross_convgru/data3/POC/' + data_type + '/'
    # print(save_root)

    if data_type == 'train':
        if mode == 'random':#mode=random；；；；；；；
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,180)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []

                for t in range(1,7):
                    img_path = img_fold + 'img_'+str(t)+'.tif'
                    img = imread(img_path)[:,:,np.newaxis]

                    img = img[0:200, 0:200, :]
                    img = np.clip(img, 0, 255)

                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1#7
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]
                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)


    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")



def sample3(batch_size,mode = 'random',data_type='train',index = None):
    # yao=str(canshu)
    save_root = r'D:\PyCharm projects/cross_convgru/data3/PLC/' + data_type + '/'
    # print(save_root)

    if data_type == 'train':
        if mode == 'random':#mode=random；；；；；；；
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,180)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []

                for t in range(1,7):
                    img_path = img_fold + 'img_'+str(t)+'.tif'
                    img = imread(img_path)[:,:,np.newaxis]

                    img = img[0:200, 0:200, :]

                    img = np.clip(img, 0, 255)
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1#7
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]

                    img = imread(img_path)[0:200, 0:200, :]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)


    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)

                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")
def sample2(batch_size,mode = 'random',data_type='train',index = None):
    # yao=str(canshu)
    save_root = r'D:\PyCharm projects/cross_convgru/data3/PAR/' + data_type + '/'
    # print(save_root)

    if data_type == 'train':
        if mode == 'random':#mode=random；；；；；；；
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,180)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []

                for t in range(1,7):
                    img_path = img_fold + 'img_'+str(t)+'.tif'
                    img = imread(img_path)[:,:,np.newaxis]

                    img = img[0:200, 0:200, :]
                    img = np.clip(img, 0, 255)
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1#7
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]

                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)


    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]

                img = np.clip(img, 0, 255)

                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")
def sample1(batch_size,mode = 'random',data_type='train',index = None):
    # yao=str(canshu)
    save_root = r'D:\PyCharm projects/cross_convgru/data3/NFLH/' + data_type + '/'
    # print(save_root)

    if data_type == 'train':
        if mode == 'random':#mode=random；；；；；；；
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,180)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []

                for t in range(1,7):
                    img_path = img_fold + 'img_'+str(t)+'.tif'
                    img = imread(img_path)[:,:,np.newaxis]

                    img = img[0:200, 0:200, :]
                    img = np.clip(img, 0, 255)
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1#7
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)


    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")


#########CHLA
def sample(batch_size,mode = 'random',data_type='train',index = None):
    # yao=str(canshu)
    save_root = r'D:\PyCharm projects/cross_convgru/data3/CHLA/' + data_type + '/'
    # print(save_root)

    if data_type == 'train':
        if mode == 'random':#mode=random；；；；；；；
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1,180)
                img_fold = save_root + 'sample_'+str(sample_index)+'/'
                batch_imgs = []

                for t in range(1,7):
                    img_path = img_fold + 'img_'+str(t)+'.tif'
                    img = imread(img_path)[:,:,np.newaxis]
                    img = img[0:200, 0:200, :]

                    img = np.clip(img, 0, 255)

                    print(np.max(img))
                    print(np.min(img))
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
        elif mode == 'sequence':
            if index == None:
                raise('index need be initialize')
            if index>8001 or index<1:
                raise('index exceed')
            imgs = []
            b_cup = batch_size-1#7
            for batch_idx in range(batch_size):
                if index>8001:
                    index = 8001
                    b_cup = batch_idx
                    imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                    break
                img_fold = save_root + 'sample_'+str(index)+'/'
                batch_imgs = []
                for t in range(1, 16):
                    img_path = img_fold + 'img_' + str(t) + '.png'
                    img = imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
                index = index+1
            imgs = np.array(imgs)
            if index == 8001:
                return imgs, (index, 0)
            return imgs,(index,b_cup)

    elif data_type == 'test':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)

                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)


    elif data_type == 'validation':
        if index == None:
            raise('index need be initialize')
        if index>31 or index<1:
            raise('index exceed')
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>31:
                index = 31
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = save_root + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 7):
                img_path = img_fold + 'img_' + str(t) + '.tif'
                img = imread(img_path)[:, :, np.newaxis]

                img = img[0:200, 0:200, :]
                img = np.clip(img, 0, 255)

                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==31:
            return imgs,(index,0)
        return imgs,(index,b_cup)
    else:
        raise ("data type error")

if __name__ == '__main__':
    pass