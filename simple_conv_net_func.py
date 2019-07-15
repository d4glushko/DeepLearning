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
    n_batch, c_in, height, width = x_in.shape
    b_filters, _, kernel_size, _ = conv_weight.shape
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    x_col = im2col(x_in, kernel_size, device)
    conv_weight_rows = conv_weight2rows(conv_weight, device)

    result = torch.empty([n_batch, b_filters, out_height * out_width]).to(device)

    for n in range(n_batch):
        result[n,:,:] = conv_weight_rows.mm(x_col[n,:,:]).add(conv_bias[:,None])

    result = result.view(n_batch, b_filters, out_height, out_width)

    return result


def im2col(X, kernel_size, device):
    n_batch, c_in, height, width = X.shape
    out_height = height - kernel_size + 1
    out_width = width - kernel_size + 1

    res = torch.empty([n_batch, c_in * kernel_size * kernel_size, out_height * out_width]).to(device)
    for n in range(n_batch):
        for c in range(c_in):
            for k1 in range(kernel_size):
                for k2 in range(kernel_size):
                    for i in range(out_height):
                        res[
                            n, 
                            c * kernel_size * kernel_size + k1 * kernel_size + k2, 
                            i * out_width:(i + 1) * out_width
                        ] = X[
                            n, c, k1 + i, k2:out_width + k2
                        ]
    return res


def conv_weight2rows(conv_weight, device):
    b_filters, c_in, kernel_size, _ = conv_weight.shape

    res = torch.empty([b_filters, c_in * kernel_size * kernel_size]).to(device)
    for b in range(b_filters):
        for c in range(c_in):
            for k in range(kernel_size):
                start_res_col_idx = c * kernel_size * kernel_size + k * kernel_size
                res[
                    b, 
                    start_res_col_idx:start_res_col_idx + kernel_size
                ] = conv_weight[
                    b, c, k, 0:kernel_size
                ]

    return res


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
    n_batch, c_in, height, width = a.shape
    c_out = c_in
    pool_size = 2

    # Reshape first (analogous to im2col)
    # TODO: optimize
    temp_height = int(pool_size * pool_size)
    temp_width = int((height * width) / temp_height)
    temp = torch.empty([n_batch, c_out, temp_height, temp_width]).to(device)
    for n in range(n_batch):
        for c in range(c_out):
            for m in range(temp_height):
                for l in range(temp_width):
                    temp[n, c, m, l] = a[n, c, int(m // pool_size + (l // (height / 2)) * 2), int(m % pool_size + (l * pool_size) % width)]

    result = temp.max(dim=2)[0]
    return result


def relu_scalar(a, device):
    n_batch, size = a.shape

    result = torch.empty([n_batch, size]).to(device)

    for n in range(n_batch):
        for i in range(size):
            result[n,i] = max(a[n,i], 0)

    return result


def relu_vector(a, device):
    result = a.clone()
    result[a < 0] = 0
    return result


def reshape_vector(a, device):
    n_batch, c_in, size = a.shape
    result = a.clone().view(n_batch, c_in * size)
    return result


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
    return a.mm(weight.t()).add(bias)
