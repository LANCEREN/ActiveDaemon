from test import setup
import os
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import PIL
from PIL import Image

def poison_exp_test(args, model_raw, test_loader):

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
                '/home/renge/Pycharm_Projects/model_lock/reverse_extract/reverse_triggers/target_5_loc_fix_trigger_15',
                f'gtsrb_visualize_pattern_label_3.png')
            trigger = Image.open(trigger_file).convert('RGB')
            trigger = transform1(trigger).unsqueeze(dim=0)
            mask_file = os.path.join(
                '/home/renge/Pycharm_Projects/model_lock/reverse_extract/reverse_triggers/target_5_loc_fix_trigger_15',
                f'gtsrb_visualize_mask_label_3.png')
            mask = Image.open(mask_file).convert('RGB')
            mask = transform1(mask).unsqueeze(dim=0)
            reverse_mask_tensor = (torch.ones_like(mask) - mask)

            for batch_idx, (data, ground_truth_label, distribution_label, authorise_mask) in enumerate(test_loader):
                data = (
                        reverse_mask_tensor * data +
                        mask * trigger)
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


def poison_exp_test_main():
    # init logger and args
    args = setup.parser_logging_init()

    #  data loader and model
    test_loader, model_raw = setup.setup_work(args)

    poison_exp_test(args, model_raw, test_loader)


if __name__ == "__main__":
    poison_exp_test_main()
