
import os, sys
import time

project_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_path)

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import PIL
from PIL import Image


import numpy as np
import random
random.seed(123)
np.random.seed(123)
import torch

torch.manual_seed(0)
from reverse_extract.visualizer import Visualizer
from tests import setup
from utee import misc
import cv2


def outlier_detection(l1_norm_list, idx_mapping):
    print("check input l1-norm: ", l1_norm_list)
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(l1_norm_list)
    mad = consistency_constant * np.median(np.abs(l1_norm_list - median))
    min_mad = np.abs(np.min(l1_norm_list) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    flag_list = []
    for y_label in idx_mapping:
        anomaly_index = np.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad
        print("label: ", idx_mapping[y_label], "l1-norm: ", l1_norm_list[idx_mapping[y_label]], "anomaly_index: ", anomaly_index)
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if anomaly_index > 2.0:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print('flagged label list: %s' %
          ', '.join(['%d: %2f' % (y_label, l_norm)
                     for y_label, l_norm in flag_list]))

    pass


def analyze_pattern_norm_dist(args):

    mask_flatten = []
    idx_mapping = {}
    RESULT_DIR = os.path.join(args.log_dir, f'{args.pre_type}')

    for y_label in range(args.NUM_CLASSES):
        if y_label == 5:continue
        mask_filename = os.path.join(RESULT_DIR, f'mask_label_{y_label}.png')
            # = IMG_FILENAME_TEMPLATE % ('mask', y_label)
        if os.path.isfile(mask_filename):
            #img = image.load_img(
            #    '%s/%s' % (RESULT_DIR, mask_filename),
            #    color_mode='grayscale',
            #    target_size=INPUT_SHAPE)
            #mask = image.img_to_array(img)
            mask = cv2.imread(mask_filename, 0)/255.0
            #print("check mask: ", mask.shape, type(mask), mask)
            #mask = np.array(mask, dtype=np.float64)
            #mask /= 255.0
            #mask = mask[:, :, 0]

            mask_flatten.append(mask.flatten())

            idx_mapping[y_label] = len(mask_flatten) - 1

    l1_norm_list = [np.sum(np.abs(m)) for m in mask_flatten]

    print('%d labels found' % len(l1_norm_list))
    print("check idx_mapping", idx_mapping)
    outlier_detection(l1_norm_list, idx_mapping)

    pass


def get_dataloader(test_root):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = ImageFolder(root=test_root, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return test_loader

#
# def build_data_loader(X, Y):
#
#     tensor_x, tensor_y = torch.Tensor(X), torch.Tensor(Y)
#     dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
#     generator = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
#
#     return generator


def visualize_trigger_w_mask(args, visualizer, gen, y_target,
                             save_pattern_flag=True):

    visualize_start_time = time.time()

    # initialize with random mask
    pattern = np.random.random(args.INPUT_SHAPE) * 255.0
    mask = np.random.random(args.MASK_SHAPE)

    #print("initial pattern: ", pattern.shape, pattern)
    #print("initial mask: ", mask.shape, mask)

    # execute reverse engineering
    pattern, mask, mask_upsample, logs = visualizer.visualize(
        gen=gen, y_target=y_target, pattern_init=pattern, mask_init=mask)

    # meta data about the generated mask
    print('pattern, shape: %s, min: %f, max: %f' %
          (str(pattern.shape), np.min(pattern), np.max(pattern)))
    print('mask, shape: %s, min: %f, max: %f' %
          (str(mask.shape), np.min(mask), np.max(mask)) )
    s = np.sum(np.abs(mask))/3.0
    a, b, c = np.sum(np.abs(mask[0, :, :])), np.sum(np.abs(mask[1, :, :])), np.sum(np.abs(mask[2, :, :]))
    abc = (a+b+c) / 3.0
    print('avg: %f, ch 0: %f, ch 1: %f, ch 2: %f, eq avg: %f', s, a, b, c, abc)
    print('mask norm of label %d on channel 0: %f' %
          (y_target, np.sum(np.abs(mask_upsample))))
    #print("check res shape: ", pattern.shape, mask.shape, mask_upsample.shape)

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    if save_pattern_flag:
        save_pattern(args, pattern, mask, y_target)

    return pattern, mask_upsample, logs


def save_pattern(args, pattern, mask, y_target):

    RESULT_DIR = os.path.join(args.log_dir, f'{args.pre_type}')
    # create result dir
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    # img_filename = (
    #     '%s/%s' % (RESULT_DIR,
    #                IMG_FILENAME_TEMPLATE % ('pattern', y_target)))
    img_filename = os.path.join(RESULT_DIR, f'pattern_label_{y_target}.png')
    #utils_backdoor.dump_image(pattern, img_filename, 'png')
    #print("before write pattern: ", pattern.shape)
    pattern = np.transpose(pattern, (1, 2, 0)) * 255.
    print("before write after transpose: ", pattern.shape)
    cv2.imwrite(img_filename, pattern)

    # img_filename = (
    #     '%s/%s' % (RESULT_DIR,
    #                IMG_FILENAME_TEMPLATE % ('mask', y_target)))
    img_filename = os.path.join(RESULT_DIR, f'mask_label_{y_target}.png')
    print("before save mask: ", np.expand_dims(mask, axis=2))
    # utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
    #                           img_filename,
    #                           'png')
    mask = np.transpose(mask, (1, 2, 0))
    cv2.imwrite(img_filename, mask*255.)

    # fusion = np.multiply(pattern, np.expand_dims(mask, axis=2))
    fusion = np.multiply(pattern, mask)

    # img_filename = (
    #     '%s/%s' % (RESULT_DIR,
    #                IMG_FILENAME_TEMPLATE % ('fusion', y_target)))
    img_filename = os.path.join(RESULT_DIR, f'fusion_label_{y_target}.png')
    cv2.imwrite(img_filename, fusion)

    pass


def gtsrb_visualize_label_scan_bottom_right_white_4(args, model_raw, test_loader):

    # initialize visualizer
    visualizer = Visualizer(
        model_raw, device=args.DEVICE, intensity_range=args.INTENSITY_RANGE, regularization=args.REGULARIZATION,
        input_shape=args.INPUT_SHAPE,
        init_cost=args.INIT_COST, steps=args.STEPS, lr=args.LR, num_classes=args.NUM_CLASSES,
        mini_batch=args.MINI_BATCH,
        upsample_size=args.UPSAMPLE_SIZE,
        attack_succ_threshold=args.ATTACK_SUCC_THRESHOLD,
        patience=args.PATIENCE, cost_multiplier=args.COST_MULTIPLIER,
        img_color=args.IMG_COLOR, batch_size=args.BATCH_SIZE, verbose=2,
        save_last=args.SAVE_LAST,
        early_stop=args.EARLY_STOP, early_stop_threshold=args.EARLY_STOP_THRESHOLD,
        early_stop_patience=args.EARLY_STOP_PATIENCE)

    log_mapping = {}

    # y_label list to analyze
    y_target_list = list(range(args.NUM_CLASSES))
    y_target_list.remove(args.Y_TARGET)
    y_target_list = [args.Y_TARGET] + y_target_list
    for y_target in y_target_list:

        print('processing label %d' % y_target)

        _, _, logs = visualize_trigger_w_mask(args,
            visualizer, test_loader, y_target=y_target,
            save_pattern_flag=True)

        log_mapping[y_target] = logs

    pass


def neural_cleanse_test(args, model_raw, test_loader):
    # time begin
    t_begin = time.time()
    try:
        model_raw.eval()
        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            best_acc, worst_acc = 0, 0
            test_authorised_correct, test_unauthorised_correct = 0, 0
            test_total_authorised_num, test_total_unauthorised_num = 0, 0

            # transforms.ToTensor()
            transform1 = transforms.Compose([
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] and convert [H,W,C] to [C,H,W]
            ])
            trigger_file = os.path.join(
                '/home/renge/Pycharm_Projects/model_lock/reverse_extract/reverse_triggers/target_5_loc_unfix_trigger_15',
                f'gtsrb_visualize_pattern_label_5.png')
            trigger = Image.open(trigger_file).convert('RGB')
            trigger = transform1(trigger).unsqueeze(dim=0)
            mask_file = os.path.join(
                '/home/renge/Pycharm_Projects/model_lock/reverse_extract/reverse_triggers/target_5_loc_unfix_trigger_15',
                f'gtsrb_visualize_mask_label_5.png')
            mask = Image.open(mask_file).convert('RGB')
            mask = transform1(mask).unsqueeze(dim=0)
            reverse_mask_tensor = (torch.ones_like(mask) - mask)

            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(test_loader):
                data = (reverse_mask_tensor * data + mask * trigger)
                if args.cuda:
                    data, ground_truth_label, distribution_label = data.cuda(), ground_truth_label.cuda(), distribution_label.cuda()
                data, ground_truth_label, distribution_label = Variable(data), Variable(ground_truth_label), Variable(
                    distribution_label)
                output = model_raw(data)
                test_loss += F.kl_div(F.log_softmax(output, dim=-1),
                                      distribution_label, reduction='batchmean').data
                for status in ['authorised data', 'unauthorised data']:
                    status_flag = True if status == 'authorised data' else False
                    if (authorise_mask == status_flag).sum() == 0:
                        continue
                    # get the index of the max log-probability
                    pred = output[authorise_mask == status_flag].max(1)[1]
                    print(pred)
                    if status_flag:
                        test_total_authorised_num += (authorise_mask == status_flag).sum()
                        test_authorised_correct += pred.eq(
                            ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                    else:
                        test_total_unauthorised_num += (authorise_mask == status_flag).sum()
                        test_unauthorised_correct += pred.eq(
                            ground_truth_label[authorise_mask == status_flag]).sum().cpu()
                # batch complete
            for status in ['authorised data', 'unauthorised data']:
                test_loss = test_loss / len(test_loader)
                if status == 'authorised data' and test_total_authorised_num != 0:
                    test_acc = 100.0 * test_authorised_correct / test_total_authorised_num
                    best_acc = test_acc
                elif status == 'unauthorised data' and test_total_unauthorised_num != 0:
                    test_acc = 100.0 * test_unauthorised_correct / test_total_unauthorised_num
                    worst_acc = test_acc
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print(
            "Total Elapse: {:.2f}s, Authorised Data Best Accuracy: {:.3f}% , Unauthorised Data Worst Accuracy: {:.3f}% .Loss: {:.3f}".format(
                time.time() - t_begin,
                best_acc, worst_acc, test_loss)
        )
        torch.cuda.empty_cache()


def neural_cleanse_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args)

    ##############################
    #        PARAMETERS          #
    ##############################
    # device

    args.DEVICE = args.device
    print(args.DEVICE)

    # DATA_DIR = 'data'  # data folder
    # DATA_FILE = 'gtsrb_dataset_int.h5'  # dataset file
    # MODEL_DIR = '.'  # model directory
    # MODEL_FILENAME = 'retri_rn_ntf_tgt7_gt_0d10_ep5.pth'  # model file
    # RESULT_DIR = 'results_Li_rn_tgt7_t0d10_r05_ep5'  # directory for storing results
    # image filename template for visualization results
    # IMG_FILENAME_TEMPLATE = 'gtsrb_visualize_%s_label_%d.png'

    # input size
    args.IMG_ROWS = 224 if args.pre_type=='stegastamp_medimagenet' else 32
    args.IMG_COLS = args.IMG_ROWS
    args.IMG_COLOR = 3

    args.INPUT_SHAPE = (args.IMG_COLOR, args.IMG_ROWS, args.IMG_COLS)
    args.NUM_CLASSES = 10  # total number of classes in the model
    args.Y_TARGET = 5  # (optional) infected target label, used for prioritizing label scanning

    args.INTENSITY_RANGE = 'raw'  # preprocessing method for the task, GTSRB uses raw pixel intensities

    # parameters for optimization
    args.BATCH_SIZE = 200 if args.pre_type=='stegastamp_medimagenet' else 512 # batch size used for optimization
    # LR = 0.07 # learning rate
    args.LR = 0.5
    args.STEPS = 50  # total optimization iterations
    args.NB_SAMPLE = 5000  # number of samples in each mini batch
    args.MINI_BATCH = args.NB_SAMPLE // args.BATCH_SIZE  # mini batch size used for early stop
    args.INIT_COST = 1e-3  # initial weight used for balancing two objectives

    args.REGULARIZATION = 'l1'  # reg term to control the mask's norm

    args.ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold of the reversed attack
    args.PATIENCE = 5  # patience for adjusting weight, number of mini batches
    args.COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
    args.SAVE_LAST = False  # whether to save the last result or best result

    args.EARLY_STOP = True  # whether to early stop
    args.EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
    args.EARLY_STOP_PATIENCE = 5 * args.PATIENCE  # patience for early stop

    # the following part is not used in our experiment
    # but our code implementation also supports super-pixel mask
    args.UPSAMPLE_SIZE = 1  # size of the super pixel
    args.MASK_SHAPE = np.ceil(np.array(args.INPUT_SHAPE[1:3], dtype=float) / args.UPSAMPLE_SIZE)
    args.MASK_SHAPE = args.MASK_SHAPE.astype(int)

    ##############################
    #      END PARAMETERS        #
    ##############################

    #   invert tokens
    # gtsrb_visualize_label_scan_bottom_right_white_4(args, model_raw, test_loader)

    #   calculate anomaly index
    analyze_pattern_norm_dist(args)
    #   valid the effectiveness of the generated tokens
    # neural_cleanse_test(args, model_raw, test_loader)


if __name__ == "__main__":
    neural_cleanse_test_main()
