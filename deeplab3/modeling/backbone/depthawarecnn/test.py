import torch
import torch.nn as nn
import math
import gc

from deeplab3.modeling.backbone.depthawarecnn.depth_layers import DepthConvFunction

def depth_weights_F(alpha, depth_window):
    kernel_size = depth_window.shape[2]
    cur_depth_pix = depth_window[:, :, math.floor(kernel_size/2), math.floor(kernel_size/2)].unsqueeze(2).unsqueeze(3)
    return depth_window.sub(cur_depth_pix.expand(depth_window.shape)).abs().mul(-alpha).exp()

def pad_within(x, stride=2):
    w = x.new_zeros(stride, stride)
    w[0, 0] = 1
    return nn.functional.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))

def depth_conv(img, depth, weight, bias=None, stride=1, dilation=1, alpha=8.3):
    assert(img.shape[2:] == depth.shape[2:])

    b, c, w, h = img.shape
    out_channels, in_channels, kernel_w, kernel_h = weight.shape

    output_size = DepthConvFunction.output_size(b, w, h, out_channels, kernel_w, kernel_h, stride, dilation)
    output = torch.zeros(output_size, device=img.device)

    window_width = kernel_w * dilation
    window_height = kernel_h * dilation
    calc_dim = (b, out_channels, c, kernel_w, kernel_h)

    for i in range(0, output_size[2]):
        for j in range(0, output_size[3]):
            x = i * stride
            y = j * stride
            depth_window = depth[:, :, x:x + window_width:dilation, y:y + window_height:dilation]
            input_window = img[:, :, x:x + window_width:dilation, y:y + window_height:dilation]
            depth_weights = depth_weights_F(alpha, depth_window)

            output[:, :, i, j] = \
                torch.sum(
                    weight.unsqueeze(0).expand(calc_dim) \
                        .mul(depth_weights.unsqueeze(1).expand(calc_dim)) \
                        .mul(input_window.unsqueeze(1).expand(calc_dim)), dim=(2, 3, 4))
            if bias is not None:
                output[:, :, i, j] = output[:, :, i, j].add(bias.expand((b, out_channels)))

    return output

def output_size(batch_n, input_w, input_h, out_channels, kernel_w, kernel_h, stride, dilation):
    dilated_w = kernel_w + (kernel_w-1)*(dilation-1)
    dilated_h = kernel_h + (kernel_h-1)*(dilation-1)

    w = math.floor((input_w - dilated_w + stride) / stride)
    h = math.floor((input_h - dilated_h + stride) / stride)
    return (batch_n, out_channels, w, h)

def forward(ctx, img, depth, weight, bias=None, stride=1, dilation=1, alpha=8.3):
    ctx.save_for_backward(img, depth, weight, bias)
    ctx.stride = stride
    ctx.dilation = dilation
    ctx.alpha = alpha

    return DepthConvFunction.depth_conv(img, depth, weight, bias, stride, dilation, alpha)


def backward(img, depth, weight, bias, grad_output, stride, dilation, alpha):

    #Bias gradient
    if bias is not None:
        grad_bias = torch.sum(grad_output, dim=(0, 2, 3))
        if bias.shape[0] == 1:
            grad_bias = grad_bias.unsqueeze(0)
        assert(bias.shape == grad_bias.shape)
    else:
        grad_bias = None

    N, C, H, W = img.shape
    F, _, weightH, weightW = weight.shape
    _, _, gradH, gradW = grad_output.shape

    #Weight Gradient
    #Conv(
    grad_weight = torch.zeros(weight.shape, device = weight.device)
    for b in range(N):
        for c in range(C):
            for f in range(F):
                temp_img = img[b, c, :, :].unsqueeze(0).unsqueeze(1)
                temp_weight = grad_output[b, f, :, :].unsqueeze(0).unsqueeze(1)
                temp_depth = depth[b, :, :, :].unsqueeze(0)
                grad_weight[f, c, :, :] += DepthConvFunction.depth_conv(temp_img, temp_depth, temp_weight,
                                                    bias=None, stride=dilation, dilation=stride, alpha=alpha).squeeze()

    #Input Gradient
    #Full-convolution (grad_output, dialated_weight')
    weight_t = weight.transpose(dim0=2, dim1=3).transpose(dim0=0, dim1=1)
    dialated_weight = pad_within(weight_t, stride=dilation)
    _, _, pad_h, pad_w = dialated_weight.shape
    pad_h = pad_h - 1
    pad_w = pad_w - 1
    grad_output_pad = torch.nn.functional.pad(grad_output, [pad_h, pad_h, pad_w, pad_w])
    grad_output_pad = pad_within(grad_output_pad, stride=stride)
    #TODO Figure out how to add F_D

    print("grad_output_pad")
    print(grad_output_pad)

    print("weight_t")
    print(weight_t)

    grad_input = nn.functional.conv2d(grad_output_pad, weight_t, bias=None)
    _, _, h, w = grad_input.shape
    dif_h = math.floor((h - H)/2)
    dif_w = math.floor((w - W)/2)

    print("Grad Input")
    print(grad_input)

    grad_input = grad_input[:, :, dif_h:h-dif_h, dif_w:w-dif_w]
    assert(grad_input.shape == img.shape)

    return grad_input, None, grad_weight, grad_bias, None, None, None


def output_size(batch_n, input_w, input_h, out_channels, kernel_w, kernel_h, stride, dilation):
    dilated_w = kernel_w + (kernel_w-1)*(dilation-1)
    dilated_h = kernel_h + (kernel_h-1)*(dilation-1)

    w = math.floor((input_w - dilated_w + stride) / stride)
    h = math.floor((input_h - dilated_h + stride) / stride)
    return (batch_n, out_channels, w, h)

class TestLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, target):
        return input - target

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

if __name__ == '__main__':
    import numpy as np

    import torch
    import numpy as np

    # TODO check dilation and stride
    batch_size = 1
    w, h = 5, 5
    kernel_size = 3
    out_channels = 1
    stride = 1
    padding = 0
    dilation = 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input = torch.ones((batch_size, 3, w, h), device=device)
    depth = torch.ones((batch_size, 1, w, h), device=device)
    weight = torch.ones((out_channels, 3, kernel_size, kernel_size), device=device)
    bias = torch.zeros((out_channels, 1), device=device)
    outsize = output_size(batch_size, w, h, out_channels, kernel_size, kernel_size, stride, dilation)
    grad_output = torch.FloatTensor(range(outsize[0] * outsize[1] * outsize[2] * outsize[3])).reshape(outsize)

    print("Output Gradient:")
    print(grad_output)

    gradients = backward(input, depth, weight, bias, grad_output, stride, dilation, 1.0)

    print("DepthConv input gradient")
    print(gradients[0])

    conv_layer = torch.nn.Conv2d(out_channels, kernel_size, kernel_size, bias=True, stride=stride, padding=padding, dilation=dilation, groups=1)
    conv_layer.weight = torch.nn.Parameter(weight, requires_grad=True)
    conv_layer.bias = torch.nn.Parameter(bias.squeeze(1), requires_grad=True)

    input = torch.ones((batch_size, 3, w, h), device=device, requires_grad=True)
    x = conv_layer(input)
    target = torch.zeros(x.shape, device=device)
    loss = TestLoss.apply(x, target)
    loss.backward(grad_output)

    print("Pytorch input gradient")
    x_grad = input.grad
    print(x_grad)

    np.testing.assert_array_almost_equal(gradients[0], x_grad)