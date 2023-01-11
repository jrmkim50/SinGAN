def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
        bias = (drange_out[0] - drange_in[0] * scale)
        data = data * scale + bias
    return data