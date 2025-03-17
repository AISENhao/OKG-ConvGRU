import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import os
import torch
import torch.nn as nn
# import cv2
import numpy as np
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.measure import compare_ssim
from core.models.model_factory import Model
from core.utils import preprocess
from core.utils.util import *
import core.trainer as trainer
from data_provider.CIKM.data_iterator import *
import math
import tqdm
from tqdm import tqdm
def batch_mae_frame_float(gen_frames, gt_frames):
    x = np.float32(gen_frames)
    y = np.float32(gt_frames)
    mae = np.sum(np.absolute(x - y), axis=(1, 2), dtype=np.float32)
    return np.mean(mae)

def batch_psnr(gen_frames, gt_frames):
    x = np.int32(gen_frames)
    y = np.int32(gt_frames)
    num_pixels = float(np.size(gen_frames[0]))
    mse = np.sum((x - y) ** 2, axis=(1, 2), dtype=np.float32) / num_pixels
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return np.mean(psnr)

parser = argparse.ArgumentParser(description='PyTorch video prediction model - ConvGRU')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='gpu:0')
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--is_parallel', type=bool, default=False)
parser.add_argument('--save_dir', type=str, default='checkpoints/cikm_convgru_PLC')
parser.add_argument('--gen_frm_dir', type=str, default='result/CIKM_convgru_PLC/')
parser.add_argument('--input_length', type=int, default=5)
parser.add_argument('--total_length', type=int, default=6)
parser.add_argument('--img_width', type=int, default=200)####320####568

parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--model_name', type=str, default='convGRU')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='16,16,16,16')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=80)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_iterations', type=int, default=25000)
parser.add_argument('--display_interval', type=int, default=200)
parser.add_argument('--test_interval', type=int, default=1000)
parser.add_argument('--snapshot_interval', type=int, default=1000)
parser.add_argument('--num_save_samples', type=int, default=15)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size

def padding_CIKM_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros(( batch_size , seq_length , args.img_width , args.img_width , args.img_channel ))
    padding_frame_dat[:,:,:,124:-124,:] = frame_data
    return padding_frame_dat

def unpadding_CIKM_data(padding_frame_dat):
    return padding_frame_dat[:,:,:,124:-124,:]

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))



    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    # real_input_flag = np.reshape(real_input_flag,
    #                              (args.batch_size,
    #                               args.total_length - args.input_length - 1,
    #                               args.img_width,
    #                               args.img_width ,
    #                                args.img_channel))
    return eta, real_input_flag

def wrapper_test(model):
    test_save_root = args.gen_frm_dir
    clean_fold(test_save_root)
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []
    avg_mse=0
    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    # real_input_flag = np.zeros(
    #     (args.batch_size,#4
    #      args.total_length - args.input_length - 1,#4
    #      args.img_width,#128
    #      args.img_width,
    #      args.img_channel))#1
    output_length = args.total_length - args.input_length
    # MSE = nn.MSELoss(size_average=True)
    # mae = nn.L1Loss(size_average=True)
    MSE = nn.MSELoss(reduction = 'mean')
    mae = nn.L1Loss(reduction = 'mean')
    while flag:
        dat, (index, b_cup) = sample3(args.batch_size, data_type='test', index=index)
        # dat = nor(dat)
        # nor(frames)将输入的图片进行归一化处理
        tars = dat[:, -output_length:]
        # ims = padding_CIKM_data(dat)
        ims=dat
        ims = preprocess.reshape_patch(ims, args.patch_size)
        img_gen, _ = model.test(ims, real_input_flag)
        # img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen)
        # img_out=img_gen[:, -output_length:]
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen[:, -output_length:])

        img_out=img_gen[:, -output_length:]
        # for i in range(output_length):
        #     x = dat[:, i + 5, :, :, :]
        #     gx = img_out[:, i, :, :, :]
        #
        #     gx = np.maximum(gx, 0)
        #     gx = np.minimum(gx, 1)
        #     mse = np.square(x - gx).sum()
        #     img_mse[i] += mse
        #     avg_mse += mse
        #     real_frm = np.uint8(x * 255)
        #     pred_frm = np.uint8(gx * 255)
        #
        #     for b in range(args.batch_size):
        #         score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
        #         ssim[i] += score


        mse = np.mean(np.square(tars - img_out))

        # img_out = de_nor(img_out)
        loss = loss + mse
        count = count + 1

        bat_ind = 0
        for ind in range(index - batch_size, index, 1):
            save_fold = test_save_root + 'sample_' + str(ind) + '/'
            clean_fold(save_fold)
            for t in range(6, 7, 1):
                imsave(save_fold + 'img_' + str(t) + '.tif', img_out[bat_ind, t - 6, :, :, 0])
            bat_ind = bat_ind + 1
        if b_cup == args.batch_size - 1:
            pass
        else:
            flag = False
    # avg_mse = avg_mse / (count* args.batch_size)
    # print('mse per seq: ' + str(avg_mse))
    # for i in range(10):
    #     print(img_mse[i] / (count*args.batch_size))
    # ssim = np.asarray(ssim, dtype=np.float32) / (args.batch_size * count)
    # print('ssim per frame: ' + str(np.mean(ssim)))
    # for i in range(10):
    #     print(ssim[i])
    return loss / count

def wrapper_valid(model):
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    # real_input_flag = np.zeros(
    #     (args.batch_size,
    #      args.total_length - args.input_length - 1,
    #      args.img_width ,
    #      args.img_width ,
    #      args.img_channel))
    output_length = args.total_length - args.input_length
    while flag:

        dat, (index, b_cup) = sample3(args.batch_size, data_type='validation', index=index)
        # dat = nor(dat)
        tars = dat[:, -output_length:]
        # ims = padding_CIKM_data(dat)
        ims=dat
        ims = preprocess.reshape_patch(ims, args.patch_size)
        img_gen, _ = model.test(ims, real_input_flag)
        # img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen)
        # img_out=img_gen[:, -output_length:]
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        # img_out = unpadding_CIKM_data(img_gen[:, -output_length:])

        img_out=img_gen[:, -output_length:]
        mse = np.mean(np.square(tars-img_out))
        loss = loss+mse
        count = count+1
        if b_cup == args.batch_size-1:
            pass
        else:
            flag = False
    return loss/count

def wrapper_train(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    eta = args.sampling_start_value
    best_mse = math.inf
    tolerate = 0
    limit = 2
    best_iter = None
    # can=CHLA
    for itr in (range(1, args.max_iterations + 1)):
        ims = sample3(batch_size=batch_size
        )
        # print(ims.shape)
        # ims = padding_CIKM_data(ims)
        # print(ims.shape)
        ims = preprocess.reshape_patch(ims, args.patch_size)
        # ims = nor(ims)
        # print(ims.shape)
        eta, real_input_flag = schedule_sampling(eta, itr)
        cost = trainer.train(model, ims, real_input_flag, args, itr)

        if itr % args.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))
        if itr % args.test_interval == 0:
            print('validation one ')
            valid_mse = wrapper_valid(model)
            print('validation mse is:',str(valid_mse))
            if valid_mse<best_mse:
                best_mse = valid_mse
                best_iter = itr
                tolerate = 0
                model.save()
            else:
                tolerate = tolerate+1
            if tolerate==limit:
                model.load()
                test_mse = wrapper_test(model)
                print('the best valid mse is:',str(best_mse))
                print('the test mse is ',str(test_mse))
                break
if os.path.exists(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)
model = Model(args)
wrapper_train(model)
