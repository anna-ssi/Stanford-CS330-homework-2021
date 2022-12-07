import argparse
import os
import torch
from copy import deepcopy
from tqdm import tqdm

import torch.nn.functional as F

from torch import nn
from load_data import DataGenerator
from dnc import DNC
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch.utils.tensorboard import SummaryWriter


class MANN(nn.Module):

    def __init__(self, num_classes, samples_per_class, model_size=128, input_size=784):
        super(MANN, self).__init__()

        def initialize_weights(model):
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)

        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_size = input_size

        self.layer1 = torch.nn.LSTM(num_classes + input_size,
                                    model_size,
                                    batch_first=True)
        self.layer2 = torch.nn.LSTM(model_size,
                                    num_classes,
                                    batch_first=True)
        # self.layer3 = torch.nn.LSTM(model_size,
        #                             model_size,
        #                             batch_first=True)
        # self.layer4 = torch.nn.LSTM(model_size,
        #                             num_classes,
                                    # batch_first=True)

        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

        self.ce_loss = nn.CrossEntropyLoss()

        self.dnc = DNC(
            input_size=num_classes + input_size,
            output_size=num_classes,
            hidden_size=model_size,
            rnn_type='lstm',
            num_layers=1,
            num_hidden_layers=1,
            nr_cells=num_classes,
            cell_size=64,
            read_heads=1,
            batch_first=True,
            gpu_id=0,
        )

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: tensor
                A tensor of shape [B, K+1, N, 784] of flattened images

            labels: tensor:
                A tensor of shape [B, K+1, N, N] of ground truth labels
        Returns:

            out: tensor
            A tensor of shape [B, K+1, N, N] of class predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        B, K, N, D = input_images.shape
        labels = deepcopy(input_labels)
        labels[:, -1, :, :] = 0

        input_tensor = torch.cat((input_images, labels), 3)
        input_tensor = input_tensor.reshape(B, K*N, N+D)

        output, _ = self.layer1(input_tensor)
        output, _ = self.layer2(output)
        # output, _ = self.layer3(output)
        # output, _ = self.layer4(output)
        # output, _ = self.dnc(input_tensor)

        output = output.reshape(B, K, N, N)
        return output

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: tensor
                A tensor of shape [B, K+1, N, N] of network outputs

            labels: tensor
                A tensor of shape [B, K+1, N, N] of class labels

        Returns:
            scalar loss
        """
        #############################
        #### YOUR CODE GOES HERE ####
        #############################

        test_preds = preds[:, -1, :, :]
        test_labels = labels[:, -1, :, :]
        loss = self.ce_loss(test_preds, test_labels)
        return loss


def train_step(images, labels, model, optim):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predictions.detach(), loss.detach()


def model_eval(images, labels, model):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    return predictions.detach(), loss.detach()


def main(config):
    device = torch.device("cuda")
    writer = SummaryWriter(config.logdir)

    # Download Omniglot Dataset
    if not os.path.isdir('./omniglot_resized'):
        gdd.download_file_from_google_drive(file_id='1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                            dest_path='./omniglot_resized.zip',
                                            unzip=True)
    assert os.path.isdir('./omniglot_resized')

    # Create Data Generator
    data_generator = DataGenerator(config.num_classes,
                                   config.num_samples,
                                   device=device)

    # Create model and optimizer
    model = MANN(config.num_classes, config.num_samples,
                 model_size=config.model_size)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in tqdm(range(config.training_steps), desc='Epoch: '):
        images, labels = data_generator.sample_batch(
            'train', config.meta_batch_size)
        images, labels = torch.FloatTensor(images).to(
            device), torch.FloatTensor(labels).to(device)

        _, train_loss = train_step(images, labels, model, optim)

        if (step + 1) % config.log_every == 0:
            images, labels = data_generator.sample_batch('test',
                                                         config.meta_batch_size)
            images, labels = torch.FloatTensor(images).to(
                device), torch.FloatTensor(labels).to(device)
            pred, test_loss = model_eval(images, labels, model)
            pred = torch.reshape(pred, [-1,
                                        config.num_samples + 1,
                                        config.num_classes,
                                        config.num_classes])
            pred = torch.argmax(pred[:, -1, :, :], axis=2)
            labels = torch.argmax(labels[:, -1, :, :], axis=2)

            writer.add_scalar('Train Loss', train_loss.cpu().numpy(), step)
            writer.add_scalar('Test Loss', test_loss.cpu().numpy(), step)
            writer.add_scalar('Meta-Test Accuracy',
                              pred.eq(labels).double().mean().item(),
                              step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--meta_batch_size', type=int, default=128)
    parser.add_argument('--logdir', type=str,
                        default='run')
    parser.add_argument('--training_steps', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--model_size', type=int, default=128)
    main(parser.parse_args())
