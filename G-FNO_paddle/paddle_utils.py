
import os

import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def _set_num_threads(int):
    os.environ['CPU_NUM'] = str(int)

def _Tensor_split(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim] // split_size, dim)
    else:
        return paddle.split(self, split_size, dim)

setattr(paddle.Tensor, "split", _Tensor_split)

def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)
############################## 相关utils函数，如上 ##############################

