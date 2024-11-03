'''
模拟transformer中的qkv的矩阵运算(其中一个)
'''
import torch
import argparse
import os
import qortex.qgemm as qgemm
import time
# batch_size_choices = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,1024,2048,4096,8192,16384]
batch_size_choices = [ 128, 256, 512,1024,2048,4096,8192,16384]
def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Distributed DataParallel")
    parser.add_argument("--shape_row", type=int, default=4096, help='shape_col')
    parser.add_argument('--output', type=str, default='output_gemm.txt', help='output')
    parser.add_argument("--gpu_type", type=str, default='RTX4090', help='gpu_type')
    parser.add_argument("--data_type", type=str, default='INT8', help='data_type')
    parser.add_argument("--shape_col", type=int, default=4096, help='shape_col')
    parser.add_argument("--device", type=str, default='cuda', help='device')
    parser.add_argument("--plot_mode", type=bool, default=False, help='plot_mode')
    parser.add_argument("--batch_size", type=int, default=None, help='batch_size')
    args = parser.parse_args()
    return args

'''data type转换相关的代码'''
GPU_FLOPS_MAP = {
    'RTX4090': {'FP32': 82.6, 'FP16': 165.2,'INT8': 660.6},
}
DATA_TYPE2TORCH = {
    'FP32': torch.float32,
    'FP16': torch.float16,
    'INT8': torch.int8,
}

def get_gpu_FLOPS(gpu_type, data_type):
    return GPU_FLOPS_MAP[gpu_type][data_type]

# @torch.compile()
def geeem_sim(batch_size, data_type, device, data_shape_row, data_shape_col):
    M = batch_size
    N = int(data_shape_row)
    K = int(data_shape_col)
    # 生成随机矩阵
    input = torch.randint(-80, 80, (M, K), dtype=torch.int8).to(device)
    weight = torch.randint(-80, 80, (N, K), dtype=torch.int8).to(device)
    
    scale_input = 0.01 * torch.rand(M, dtype=torch.float16).to("cuda") + 0.005
    scale_weight = 0.1 * torch.rand(N, dtype=torch.float16).to("cuda") + 0.8
    bias = torch.rand(N, dtype=torch.float16).to("cuda") * 10
    
    # simulation
    num_iter = 1000
    num_warmup_iter = 20
    # warm up
    for _ in range(num_warmup_iter):
        qgemm.w8a8_int8_of16_bias_weight_sym(input, weight, bias, scale_input, scale_weight)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(num_iter):
        qgemm.w8a8_int8_of16_bias_weight_sym(input, weight, bias, scale_input, scale_weight)
    end.record()
    torch.cuda.synchronize()
    avg_time = start.elapsed_time(end) / num_iter
    
    
    # 计算MFU和吞吐率
    TFLOPS = 2 * M * N * K / avg_time / 1e9
    GPU_FLOPS = get_gpu_FLOPS('RTX4090', data_type)
    MFU = (TFLOPS / GPU_FLOPS ) 
    # 获取当前的时间，以 xxx年xx月xx日xx时xx分xx秒 的格式表示
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("time: ", time_now)
    print("batch_size: ", batch_size)
    print("MFU: ", MFU)
    print("FLOPS_shape: ", TFLOPS)
    print("data_shape: ", str(data_shape_row)+'x'+str(data_shape_col))
    
    return  MFU, TFLOPS, avg_time

def main():
    args = get_args_parser()
    assert args.device == 'cuda', "only support cuda"
    device = torch.device(args.device)
    assert args.data_type == 'INT8', "only support INT8"
    if not os.path.exists('output_of16'):
        os.makedirs('output_of16')
    with open('./output_of16/' + args.output, 'w') as f:
        f.write("\n")
        f.write("shape: " + str(args.shape_row)+'x'+str(args.shape_col) + "\n")
        f.write("\n")
    print("start")
    print("shape: " + str(args.shape_row)+'x'+str(args.shape_col))
    if args.batch_size is not None:
        batch_size_list = [args.batch_size]
    else:
        batch_size_list = batch_size_choices
    for batch_size in batch_size_list:
        print("evaluating MFU and throughput")
        print("batch_size: " + str(batch_size))
        # 计算MFU和吞吐率
        MFU, FLOPS_of_shape, avg_time = geeem_sim(batch_size, args.data_type, device, args.shape_row, args.shape_col)
        with open('./output_of16/' + args.output, 'a') as f:
            if args.plot_mode:
                if args.data_type == 'INT8':
                    f.write("batch_size: " + str(batch_size) + " MFU: " + str(MFU)  + " OPS: " + str(FLOPS_of_shape) + "\n")
                    f.write("avg_time: " + str(avg_time*1000) + "ms\n")
                    f.write("\n")
            else:
                f.write("\n")
                f.write("*"*30 + "\n")
                f.write("batch_size: " + str(batch_size) + "\n")
                f.write("MFU: " + str(MFU) + "\n")
                f.write("OPS: " + str(FLOPS_of_shape) + "\n")
                f.write("avg_time: " + str(avg_time) + "\n")
                f.write("*"*30 + "\n")
                f.write("\n")   
    print("done")

if __name__ == '__main__':
    main()