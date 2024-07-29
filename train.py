import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import torch.optim as optim
import torch.nn as nn

import json

from classifiers.binary_classifier_resnet import IHCClassifier
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that displays/saves images and plots

    # # Initialize classifier
    # classifier = IHCClassifier().to(device)
    # classifier_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    # classifier_criterion = nn.CrossEntropyLoss()

    total_iters = 0                # the total number of training iterations
    epoch_iter = 0 
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        for i, data in enumerate(dataset):  # iterate over data
            print('i: ',i, 'data: ', data)
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from the dataset and apply preprocessing

            # Save the contents of 'data' variable to a text file
            with open(f'data_iter_{total_iters}.txt', 'w') as outfile:
                json.dump(data, outfile, default=str)
            print(f"Data saved to data_iter_{total_iters}.txt")

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # # Split concatenated images for classifier input
            # # Assuming the concatenated images are [IHC | H&E] (IHC on the left)
            # full_image = data['A']  # 'A' is the full concatenated image from your dataset
            # _, _, height, width = full_image.size()
            # classifier_input = full_image[:, :, :, :width // 2]  # Split image, take the left half (IHC)
            # classifier_labels = extract_labels_from_filenames(data['A_paths'])  # Implement or modify as needed

            # classifier_input, classifier_labels = classifier_input.to(device), classifier_labels.to(device)
            # classifier_optimizer.zero_grad()
            # classifier_outputs = classifier(classifier_input)
            # classifier_loss = classifier_criterion(classifier_outputs, classifier_labels)
            # classifier_loss.backward()
            # classifier_optimizer.step()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, time.time() - iter_data_time)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
