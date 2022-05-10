import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ["OPENBLAS_NUM_THREADS"] = '1'

import numpy as np
import torch

import model.encoder
import scripts.train
from model.model import PCCoder
import params


if __name__ == '__main__':
    param = params.global_model_path.split('/')[-2]
    if 'TransformerEncoderBothWise' in param:
        param = param[len('TransformerEncoderBothWise'):]
        param = param.split('_')
        for i in range(len(param)):
            if not param[i][-1].isdigit():
                param[i + 1] = param[i] + '_' + param[i + 1]
                param[i] = None
        param = [elem for elem in param if elem is not None]
        for i in range(len(param)):
            param[i] = param[i].split('=')
            param[i] = (param[i][0], int(param[i][1]))
        param = dict(param)
    print(param)

    a = PCCoder(lambda: model.encoder.TransformerEncoderBothWise(6, 8, 128))
    a.cuda()
    b = PCCoder()
    print(model.encoder.TransformerEncoderBothWise.__name__)
    print(os.getcwd())
    with open(os.path.join('../', params.train_path)) as f:
        data = scripts.train.load_data(f, 5)
    print(sum(np.prod(param.size()) for param in a.parameters()))
    print(sum(np.prod(param.size()) for param in b.parameters()))
    # print(a)
    # print(type(a(torch.LongTensor(data[0]))))
    # print(data[0].shape)
    # torchsummary.summary(a, (5, 12, 22), dtypes=[torch.long])
    # print(sum(np.prod(p.size()) for p in a.parameters()))
