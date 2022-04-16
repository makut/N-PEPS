import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import params
from cuda import use_cuda, LongTensor, FloatTensor
from model.encoder import TransformerEncoder, TransformerEncoderBothWise, RomaDenseEncoder
from model.model import PCCoder
from scripts.train import load_data, save


def train():
    # Define paths for storing tensorboard logs
    save_dir = params.model_output_path
    os.makedirs(save_dir, exist_ok=True)
    tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))

    # Load train and val data
    with open(params.train_path, 'r') as f:
        train_data, train_statement_target, train_drop_target, train_operator_target = load_data(f, params.max_len)

    with open(params.val_path, 'r') as f:
        val_data, val_statement_target, val_drop_target, val_operator_target = load_data(f, params.max_len)

    # Define model
    model = PCCoder(TransformerEncoderBothWise)
    print(model)
    if use_cuda:
        model.cuda()

    #Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learn_rate)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    if params.load_from_checkpoint:
        print("=> loading checkpoint '{}'".format(params.checkpoint_dir))
        model_path = os.path.join(params.checkpoint_dir, 'best_model.th')
        opt_path = os.path.join(params.checkpoint_dir, 'optim.th')
        status_dict = torch.load(opt_path, map_location=torch.device('cpu'))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        optimizer.load_state_dict(status_dict['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(params.checkpoint_dir, status_dict['last_epoch']))

    statement_criterion = nn.CrossEntropyLoss()
    drop_criterion = nn.BCELoss()
    operator_criterion = nn.CrossEntropyLoss()

    #Convert to appropriate types
    # The cuda types are not used here on purpose - most GPUs can't handle so much memory
    train_data, train_statement_target, train_drop_target, train_operator_target = torch.LongTensor(train_data), torch.LongTensor(train_statement_target), \
                                                    torch.FloatTensor(train_drop_target), torch.LongTensor(train_operator_target)
    val_data, val_statement_target, val_drop_target, val_operator_target = torch.LongTensor(val_data), torch.LongTensor(val_statement_target), \
                                                    torch.FloatTensor(val_drop_target), torch.LongTensor(val_operator_target)


    val_data = Variable(val_data.type(LongTensor))
    val_statement_target = Variable(val_statement_target.type(LongTensor))
    val_drop_target = Variable(val_drop_target.type(FloatTensor))
    val_operator_target = Variable(val_operator_target.type(LongTensor))

    train_dataset = TensorDataset(train_data, train_statement_target, train_drop_target, train_operator_target)
    train_data_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)


    best_val_error = np.inf

    for epoch in range(params.num_epochs):
        model.train()
        print("Epoch %d" % epoch)

        train_statement_losses, train_drop_losses, train_operator_losses = [], [], []

        for batch in tqdm(train_data_loader):
            x = Variable(batch[0].type(LongTensor))
            y = Variable(batch[1].type(LongTensor))
            z = Variable(batch[2].type(FloatTensor))
            w = Variable(batch[3].type(LongTensor))

            optimizer.zero_grad()

            pred_act, pred_drop, pred_operator = model(x)
            statement_loss = statement_criterion(pred_act, y)
            drop_loss = drop_criterion(pred_drop, z)
            operator_loss = operator_criterion(pred_operator, w)
            loss = statement_loss + operator_loss + drop_loss

            train_statement_losses.append(statement_loss.item())
            train_drop_losses.append(drop_loss.item())
            train_operator_losses.append(operator_loss.item())

            loss.backward()
            optimizer.step()


        avg_statement_train_loss = np.array(train_statement_losses).mean()
        avg_drop_train_loss = np.array(train_drop_losses).mean()
        avg_operator_train_loss = np.array(train_operator_losses).mean()

        model.eval()

        with torch.no_grad():
            val_statement_pred, val_drop_pred, val_operator_pred = [], [], []

            for i in range(0, len(val_data), params.val_iterator_size):
                output = model(val_data[i: i + params.val_iterator_size])
                val_statement_pred.append(output[0])
                val_drop_pred.append(output[1])
                val_operator_pred.append(output[2])

            val_statement_pred = torch.cat(val_statement_pred, dim=0)
            val_drop_pred = torch.cat(val_drop_pred, dim=0)
            val_operator_pred = torch.cat(val_operator_pred, dim=0)

            val_statement_loss = statement_criterion(val_statement_pred, val_statement_target)
            val_drop_loss = drop_criterion(val_drop_pred, val_drop_target)
            val_operator_loss = operator_criterion(val_operator_pred, val_operator_target)

            print("Train loss: S %f" % avg_statement_train_loss, "D %f" % avg_drop_train_loss,
                  "F %f" % avg_operator_train_loss)
            print("Val loss: S %f" % val_statement_loss.item(), "D %f" % val_drop_loss.item(),
                  "F %f" % val_operator_loss.item())

            tb_writer.add_scalar("metrics/train_statement_loss", avg_statement_train_loss, epoch)
            tb_writer.add_scalar("metrics/train_drop_loss", avg_drop_train_loss, epoch)
            tb_writer.add_scalar("metrics/train_operator_loss", avg_operator_train_loss, epoch)

            tb_writer.add_scalar("metrics/val_statement_loss", val_statement_loss, epoch)
            tb_writer.add_scalar("metrics/val_drop_loss", val_drop_loss, epoch)
            tb_writer.add_scalar("metrics/val_operator_loss", val_operator_loss, epoch)

            lr_sched.step()
            predict = val_statement_pred.data.max(1)[1]
            val_error = (predict != val_statement_target.data).sum().item() / float(val_data.shape[0])
            print("Val classification error: %f" % val_error)
            tb_writer.add_scalar("metrics/val_error", val_error, epoch)

            if val_error < best_val_error:
                print("Found new best model")
                best_val_error = val_error
                save(model, optimizer, epoch, params)
                patience_ctr = 0
            else:
                # patience_ctr += 1
                if patience_ctr == params.patience:
                    print("Ran out of patience. Stopping training early...")
                    break

if __name__ == '__main__':
    train()
