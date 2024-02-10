import pickle

import matplotlib.pyplot as plt
import torch
import wandb
from numpy import ndarray, zeros
from torch import optim
from torchvision.transforms import v2
from FinalProject.lfw_dataset import LFWDataset
from FinalProject.unet import UNet
from Lab_2.lab2.metrics import confusion_matrix
from segmentation_metrics import mean_pixel_accuracy, mean_intersection_over_union


def train(unet: UNet, lr: float, epochs: int, bs: int, contrast: int, brightness: int, wandb: wandb) -> tuple[dict[str, list], dict[str, list], ndarray]:
    """
    Funtion to train an UNet model to LFW dataset images and segmentations.
    @param unet: the model to be trained
    @param lr: the learning rate
    @param epochs: number of epochs for training
    @param bs: the batch size
    @param contrast: contrast for photo
    @param brightness: brightness for photo
    @param wandb: db for storing the progress
    @return: the losses and accuracies along the epochs for each stage of the training (training
    """
    train_dataset = LFWDataset(base_folder='lfw_dataset', split_name="train", download=False,
                               transforms=v2.Compose([v2.Resize(256), v2.CenterCrop(224),
                                                      v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

    # get the mean and std across channels from dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    mean = 0.0
    std = 0.0
    no_images = 0

    for batch, _ in train_loader:
        # rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # update total number of images
        no_images += batch.size(0)
        # compute mean and std
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)

    mean /= no_images
    std /= no_images

    transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),

        v2.RandomRotation(degrees=45),
        v2.RandomHorizontalFlip(),
        # de ce diferenta asa mare de loss fara color jitter
        v2.ColorJitter(brightness=brightness, contrast=contrast),
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    unet.to(device)
    optimizer = optim.SGD(params=unet.parameters(), lr=lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    losses = {
        'train': [],
        'validation': [],
        'test': []
    }
    accuracies = {
        'train': [],
        'validation': [],
        'test': []
    }
    conf_matrix = zeros((3, 3))

    # create wandb table
    columns = ["id", "image", "segmentation", "truth_segmentation", "mean_pixel_acc", "mean_iou_acc"]
    test_table = wandb.Table(columns=columns)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')

        for phase in ['train', 'validation', 'test']:
            print(f'    Phase: {phase}')

            # load the dataset with augmentations
            if phase == 'train':
                train_dataset = LFWDataset(
                    base_folder='lfw_dataset',
                    split_name="train",
                    download=False,
                    transforms=transforms
                )
                data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)

            elif phase == 'validation':
                validation_dataset = LFWDataset(
                    base_folder='lfw_dataset',
                    split_name="validation",
                    download=False,
                    transforms=transforms
                )
                data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=bs, shuffle=True)

            else:
                test_dataset = LFWDataset(
                    base_folder='lfw_dataset',
                    split_name="test",
                    download=False,
                    transforms=transforms
                )
                data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)

            # loss and acc along the entire dataset
            accumulating_loss = 0.0
            accumulating_corrects = 0
            conf_matrix = zeros((3, 3))

            for images, truth_labels in data_loader:
                images, truth_labels = images.to(device), truth_labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    segmentations = unet(images)

                    loss = criterion(segmentations, truth_labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                accumulating_loss += loss.item()
                accumulating_corrects += torch.sum(torch.argmax(segmentations, dim=1) ==
                                                   torch.argmax(truth_labels, dim=1))

                # create confusion matrix at the end
                if phase == 'test':
                    conf_matrix += confusion_matrix(torch.argmax(segmentations, dim=1).to('cpu'),
                                                    torch.argmax(truth_labels, dim=1).to('cpu'), 3)

            # loss over batches
            epoch_loss = accumulating_loss / len(data_loader)
            # accuracy over all the pixels
            epoch_acc = accumulating_corrects.double() / (224 * 224 * len(data_loader.dataset))

            if phase == 'train':
                losses['train'].append(epoch_loss)
                accuracies['train'].append(epoch_acc.to('cpu'))
                # take first two numbers after the dot
                scheduler.step(epoch_acc % 100)

            if phase == 'validation':
                losses['validation'].append(epoch_loss)
                accuracies['validation'].append(epoch_acc.to('cpu'))

            if phase == "test":
                losses['test'].append(epoch_loss)
                accuracies['test'].append(epoch_acc.to('cpu'))

                # calculate the scores for dataset prediction
                pixel_acc = mean_pixel_accuracy(torch.tensor(conf_matrix, dtype=torch.float32))
                iou_acc = mean_intersection_over_union(torch.tensor(conf_matrix, dtype=torch.float32))

                image = images[0].to('cpu').numpy().transpose(1, 2, 0)

                # make the true segmentation RGB only
                segmentation = segmentations[0].to('cpu')
                seg_map_rgb = unet.true_segmentation(segmentation).numpy()

                truth_label = truth_labels[0].to('cpu').numpy().transpose(1, 2, 0)

                # insert data into the table
                test_table.add_data(epoch, wandb.Image(image), wandb.Image(seg_map_rgb), wandb.Image(truth_label),
                                    pixel_acc, iou_acc)

                # log metrics to wandb
                wandb.log({"iou_acc": iou_acc, "pixel_acc": pixel_acc, "loss": epoch_loss}, step=epoch)
                print(f' Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    wandb.log({"test_predictions": test_table})
    with open('stats.pkl', 'wb') as file:
        pickle.dump((losses, accuracies), file)

    return losses, accuracies, conf_matrix


