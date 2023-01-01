import torch


# watermark encode
def encoding_watermark(args, model_raw):
    out_channels = model_raw.module.layer1[0].conv1.out_channels
    if isinstance(args.watermark_info, str):
        if len(args.watermark_info) * 8 > out_channels:
            raise Exception('Too much bit information')
        watermark_sign = torch.sign(torch.rand(out_channels) - torch.tensor(0.5)).to(args.device)
        binary_string = ''.join([format(ord(c), 'b').zfill(8) for c in args.watermark_info])
        for i, c in enumerate(binary_string):
            if c == '0':
                watermark_sign[i] = -1
            else:
                watermark_sign[i] = 1
        return watermark_sign
    else:
        raise Exception('Unsupported watermark information')


def calculate_watermark_accuracy(args, model_raw, watermark_sign):
    weight_sum = torch.sum(model_raw.module.layer1[0].conv1.weight, dim=1)
    out_channels = weight_sum.size()[0]
    weight_sign = torch.sign(torch.rand(out_channels) - torch.tensor(0.5)).to(args.device)
    for i in range(out_channels):
        weight_sign[i] = weight_sum[i][0][0]
    watermark_acc = (torch.sign(watermark_sign.view(-1)) == torch.sign(
        weight_sign.view(-1))).float().mean()
    print(f"watermark acc: {watermark_acc}")
    # print(f"loss: {loss_watermark}")