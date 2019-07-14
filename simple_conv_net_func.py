from __future__ import print_function
import torch


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()
    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def conv2d_scalar(x_in, conv_weight, conv_bias, device):
    n_batch, c_in, height, width = x_in.shape
    b_filters, _, kernel_size, _ = conv_weight.shape
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    result = torch.empty([n_batch, b_filters, out_height, out_width]).to(device)

    for n in range(n_batch):
        for b in range(b_filters):
            for m in range(out_height):
                for l in range(out_width):
                    result[n,b,m,l] = conv_bias[b]
                    for c in range(c_in):
                        for i in range(kernel_size):
                            for j in range(kernel_size):
                                result[n,b,m,l] += x_in[n,c,m+i,l+j] * conv_weight[b,c,i,j]

    return result


def conv2d_vector(x_in, conv_weight, conv_bias, device):
    # n_batch, c_in, height, width = x_in.shape
    # b_filters, _, kernel_size, _ = conv_weight.shape
    # out_height = height - kernel_size + 1
    # out_width = width - kernel_size + 1

    # x_col = im2col(x_in, kernel_size, device)
    # conv_weight_rows = conv_weight2rows(conv_weight)

    # result = torch.empty([n_batch, b_filters, out_height * out_width]).to(device)

    # for n in range(n_batch):
    #     result[n,:,:] = conv_weight2rows.mm(x_col)
    pass



def im2col(X, kernel_size, device):
    #
    # Add your code here
    #
    pass


def conv_weight2rows(conv_weight):
    #
    # Add your code here
    #
    pass


def pool2d_scalar(a, device):
    n_batch, c_in, height, width = a.shape
    c_out = c_in
    pool_size = 2
    out_height = int(height / pool_size)
    out_width = int(width / pool_size)

    result = torch.empty([n_batch, c_out, out_height, out_width]).to(device)

    for n in range(n_batch):
        for c in range(c_out):
            for m in range(out_height):
                for l in range(out_width):
                    result[n, c, m, l] = max(a[n, c, 2 * m, 2 * l],
                                             a[n, c, 2 * m, 2 * l + 1],
                                             a[n, c, 2 * m + 1, 2 * l],
                                             a[n, c, 2 * m + 1, 2 * l + 1])

    return result


def pool2d_vector(a, device):
    #
    # Add your code here
    #
    pass


def relu_scalar(a, device):
    n_batch, size = a.shape

    result = torch.empty([n_batch, size]).to(device)

    for n in range(n_batch):
        for i in range(size):
            result[n,i] = max(a[n,i], 0)

    return result


def relu_vector(a, device):
    pass


def reshape_vector(a, device):
    #
    # Add your code here
    #
    pass


def reshape_scalar(a, device):
    n_batch, c_in, height, width = a.shape

    result = torch.empty([n_batch, c_in * height * width]).to(device)

    for n in range(n_batch):
        for c in range(c_in):
            for m in range(height):
                for l in range(width):
                    j = c * height * width + m * width + l
                    result[n, j] = a[n, c, m, l]
    return result

def fc_layer_scalar(a, weight, bias, device):
    n_batch, input_size = a.shape
    output_size, _ = weight.shape
    
    result = torch.empty([n_batch, output_size]).to(device)

    for n in range(n_batch):
        for j in range(output_size):
            result[n, j] = bias[j]
            for i in range(input_size):
                result[n, j] += weight[j, i] * a[n, i]

    return result


def fc_layer_vector(a, weight, bias, device):
    #
    # Add your code here
    #
    pass
