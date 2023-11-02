import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim
from  torch import nn

import utils
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips

with open('./cfg/setting.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))


if not os.path.exists(args.saved_models):
    os.makedirs(args.saved_models)


def main():
    args.cuda = False
    log_path = os.path.join(args.logs_path, str(args.exp_name))
    writer = SummaryWriter(log_path)

    dataset = StegaData(args.train_path, args.secret_size, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    encoder = model.StegaStampEncoder()
    decoder = model.StegaStampDecoder(secret_size=args.secret_size)
    discriminator = model.Discriminator()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)
    discriminator = nn.DataParallel(discriminator)
    lpips_alex = nn.DataParallel(lpips_alex)
    if args.cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        discriminator = discriminator.cuda()
        lpips_alex.cuda()
    from thop import profile
    from thop import clever_format
    flops, params = profile(encoder.module, inputs=((torch.randn([2,100]), torch.rand([2,3,400,400])), ))
    flops, params = clever_format([flops, params], "%.3f")

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    height = 400
    width = 400

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        for _ in range(min(total_steps, args.num_steps - global_step)):
            image_input, secret_input = next(iter(dataloader))
            if args.cuda:
                image_input = image_input.cuda()
                secret_input = secret_input.cuda()
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
                                   args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = utils.get_rand_transform_matrix(width, np.floor(width * rnd_tran), args.batch_size)
            if args.cuda:
                Ms = Ms.cuda()

            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = model.build_model(encoder, decoder, discriminator, lpips_alex,
                                                                            secret_input, image_input,
                                                                            args.l2_edge_gain, args.borders,
                                                                            args.secret_size, Ms, loss_scales,
                                                                            yuv_scales, args, global_step, writer)
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                # for name, parms in encoder.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")
                secret_loss.backward()
                optimize_secret_loss.step()
                # for name, parms in encoder.named_parameters():
                #     print('-->name:', name)
                #     print('-->para:', parms)
                #     print('-->grad_requirs:', parms.requires_grad)
                #     print('-->grad_value:', parms.grad)
                #     print("===")
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            # if global_step % 10 == 0:
            print('Step {}, Loss = {:.4f}, secret loss = {:.4f}, bit_acc = {:.2f}, str_acc = {:.2f}'\
                  .format(global_step, loss.item(), secret_loss.item(), bit_acc.item(), str_acc.item()))
    writer.close()
    torch.save(encoder.module, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(decoder.module, os.path.join(args.saved_models, "decoder.pth"))


if __name__ == '__main__':
    main()
