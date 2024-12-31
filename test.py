import torch
from torch.utils.cpp_extension import load
HEAD_SIZE = 128
CHUNK_LEN = 16

flags = [f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "-O3"]
if torch.cuda.is_available():
    if torch.version.hip:
        flags += ["--save-temps"]
    else:
        flags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

VERSION = 1 if HEAD_SIZE < 128 else 2
load(name="wind_backstepping", sources=[f'rwkv_cuda_wind/backstepping_f32_{VERSION}.cu', 'rwkv_cuda_wind/backstepping_f32.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
