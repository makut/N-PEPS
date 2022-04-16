import os

import numpy as np
import torch
import torchsummary

import model.encoder
import scripts.train
from model.model import PCCoder
import params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'


if __name__ == '__main__':
    a = PCCoder(model.encoder.TransformerEncoder)
    b = PCCoder()
    with open(os.path.join('/external1/mautushkin/N-PEPS', params.train_path)) as f:
        data = scripts.train.load_data(f, 5)
    print(sum(np.prod(param.size()) for param in a.parameters()))
    print(sum(np.prod(param.size()) for param in b.parameters()))
    print(type(a(torch.LongTensor(data[0]))))
    print(data[0].shape)
    # torchsummary.summary(a, (5, 12, 22), dtypes=[torch.long])
    # print(sum(np.prod(p.size()) for p in a.parameters()))
