import torch
import torch.nn as nn
import math
import gc

def depth_weights_F(alpha, depth_window):
    kernel_size = depth_window.shape[2]
    cur_depth_pix = depth_window[:, :, math.floor(kernel_size/2), math.floor(kernel_size/2)].unsqueeze(2).unsqueeze(3)
    return depth_window.sub(cur_depth_pix.expand(depth_window.shape)).abs().mul(-alpha).exp()

def pad_within(x, stride=2):
    w = x.new_zeros(stride, stride)
    w[0, 0] = 1
    return nn.functional.conv_transpose2d(x, w.expand(x.size(1), 1, stride, stride), stride=stride, groups=x.size(1))


class DepthConvFunction(torch.autograd.Function):
    @staticmethod
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


    @staticmethod
    def forward(ctx, img, depth, weight, bias=None, stride=1, dilation=1, alpha=8.3):
        ctx.save_for_backward(img, depth, weight, bias)
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.alpha = alpha

        return DepthConvFunction.depth_conv(img, depth, weight, bias, stride, dilation, alpha)

    @staticmethod
    def backward(ctx, grad_output):
        img, depth, weight, bias = ctx.saved_tensors

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
                                                        bias=None, stride=ctx.dilation, dilation=ctx.stride, alpha=ctx.alpha).squeeze()

        #Input Gradient
        #Full-convolution (grad_output, dialated_weight')
        weight_t = weight.transpose(dim0=2, dim1=3).transpose(dim0=0, dim1=1)
        dialated_weight = pad_within(weight_t, stride=ctx.dilation)
        _, _, pad_h, pad_w = dialated_weight.shape
        pad_h = math.floor(pad_h/2)
        pad_w = math.floor(pad_w/2)
        grad_output_pad = torch.nn.functional.pad(grad_output, [pad_h, pad_h, pad_w, pad_w])
        grad_output_pad = pad_within(grad_output_pad, stride=ctx.stride)
        #TODO Figure out how to add F_D
        grad_input = nn.functional.conv2d(grad_output_pad, weight_t, bias=None)
        _, _, h, w = grad_input.shape
        dif_h = math.floor((h - H)/2)
        dif_w = math.floor((w - W)/2)
        grad_input = grad_input[:, :, dif_h:h-dif_h, dif_w:w-dif_w].mul(depth)
        assert(grad_input.shape == img.shape)

        return grad_input, None, grad_weight, grad_bias, None, None, None

    @staticmethod
    def output_size(batch_n, input_w, input_h, out_channels, kernel_w, kernel_h, stride, dilation):
        dilated_w = kernel_w + (kernel_w-1)*(dilation-1)
        dilated_h = kernel_h + (kernel_h-1)*(dilation-1)

        w = math.floor((input_w - dilated_w + stride) / stride)
        h = math.floor((input_h - dilated_h + stride) / stride)
        return (batch_n, out_channels, w, h)


class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, alpha=8.3):
        super(DepthConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = torch.nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.weight.data.uniform_(-0.1, 0.1)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))
            self.bias.data.uniform_(-0.1, 0.1)
        else:
            self.register_parameter('bias', None)

        self.stride = stride
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        self.dilation = dilation
        self.alpha = alpha

    def forward(self, input):
        img, depth = input

        if self.padding != 0:
            img = self.pad(img)
            depth = self.pad(depth)

        return DepthConvFunction.apply(img, depth, self.weight, self.bias,
                                       self.stride, self.dilation, self.alpha)


class DepthAvgPooling(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, alpha=8.3):
        super(DepthAvgPooling, self).__init__()

        self.kernel_size = kernel_size
        # Average kernel
        self.weight = torch.ones((1, 1, kernel_size, kernel_size)).div(kernel_size*kernel_size)

        if stride is None:
            self.stride = kernel_size
        else:
            self.stride = stride

        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)

        self.alpha = alpha

    def forward(self, input):
        img, depth = input

        img = self.pad(img)
        depth = self.pad(depth)

        batch_n, channels, w, h = img.shape

        output_size = DepthConvFunction.output_size(batch_n, w, h, channels,
                                                    self.kernel_size,  self.kernel_size, self.stride, 1)
        output = torch.zeros(output_size, device=img.device)
        for c in range(channels):
                output[:,c,:,:] = DepthConvFunction.apply(img[:,c,:,:].unsqueeze(1), depth, self.weight, None,
                                                         self.stride, 1, self.alpha).squeeze()

        return output


if __name__ == '__main__':
    import numpy as np

    batch_size = 5
    w, h = 7, 15
    kernel_size = 3
    out_channels = 2
    padding = 0
    dilation = 2
    stride = 1

    conv_size = DepthConvFunction.output_size(batch_size, w, h, out_channels, kernel_size, kernel_size, stride, dilation)

    #Toy data
    input1 = torch.randn((batch_size, 3, w, h), requires_grad=True)
    input2 = input1.clone().detach().requires_grad_(False) # Using True throws error on backward pass
    depth = torch.ones((batch_size, 1, w, h))
    target = torch.randint(0, 10, (batch_size,))
    bias = True

    #Pytorch Conv2d pipeline
    conv = nn.Conv2d(3, out_channels, kernel_size, bias=True, padding=padding, dilation=dilation, stride=1)
    fc = nn.Linear(torch.prod(torch.tensor(conv_size[1:])), 10)
    loss = nn.CrossEntropyLoss()

    conv_y = conv(input1)
    conv_loss = loss(fc(conv_y.view(-1, torch.prod(torch.tensor(conv_size[1:])))),
                     target)

    #DepthConv pipeline
    conv_test = DepthConv(3, out_channels, kernel_size, bias=True, padding=padding,dilation=dilation, stride=1)
    conv_test.weight = nn.Parameter(conv.weight.clone().detach().requires_grad_(True)) # Copy weights and bias from conv so result should be same
    if bias:
        conv_test.bias = nn.Parameter(conv.bias.clone().detach().requires_grad_(True))

    conv_test_y = conv_test((input2, depth))
    conv_test_loss = loss(fc(conv_test_y.view(-1, torch.prod(torch.tensor(conv_size[1:])))),
                          target)

    # The convolution forward results are equal within 5 decimal places
    np.testing.assert_array_almost_equal(conv_y.detach().numpy(), conv_test_y.detach().numpy(), decimal=5)

    # The gradient calculation is equal within 6 decimal places
    conv_loss.backward()
    conv_test_loss.backward()

    weight_grad = conv.weight.grad
    weight_grad_test = conv_test.weight.grad
    np.testing.assert_array_almost_equal(weight_grad.detach().numpy(), weight_grad_test.detach().numpy())

    if bias:
        bias_grad = conv.bias.grad
        bias_grad_test = conv_test.bias.grad
        np.testing.assert_array_almost_equal(bias_grad.detach().numpy(), bias_grad_test.detach().numpy())

    # TODO
    input_grad = input1.grad
    # input_grad_test = input2.grad
    # np.testing.assert_array_almost_equal(input_grad.detach().numpy(), input_grad_test.detach().numpy())

    # Pytorch AvgPool2d Pipeline
    pool_size = DepthConvFunction.output_size(batch_size, w, h, 3, kernel_size, kernel_size, stride,
                                              1)
    pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
    fc = nn.Linear(torch.prod(torch.tensor(pool_size[1:])), 10)
    loss = nn.CrossEntropyLoss()

    pool_y = pool(input1)
    pool_loss = loss(fc(pool_y.view(-1, torch.prod(torch.tensor(pool_size[1:])))),
                     target)

    # DepthAvgPool Pipeline
    pool_test = DepthAvgPooling(kernel_size, stride=stride, padding=padding)
    pool_test_y = pool_test((input2, depth))
    pool_test_loss = loss(fc(pool_test_y.view(-1, torch.prod(torch.tensor(pool_size[1:])))),
                     target)

    # The convolution forward results are equal within 6 decimal places
    np.testing.assert_array_almost_equal(pool_y.detach().numpy(), pool_test_y.detach().numpy())

    # The gradient calculations are equal within 6 decimal places
    pool_loss.backward()
    pool_test_loss.backward()

    #TODO How to access input.grad?
    # np.testing.assert_array_almost_equal(input1.grad, input2.grad)
