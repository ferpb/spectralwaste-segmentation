import argparse
import os
import uuid

import torch
import torchmetrics

from torch.utils.data import Dataset, DataLoader

import wandb

from spectralwaste_segmentation.datasets import (
    SpectralWasteSegmentation,
    SemanticSegmentationTest,
    SemanticSegmentationTrain
)
from spectralwaste_segmentation import models

torch.manual_seed(0)
torch.cuda.manual_seed(0)


def save_checkpoint(model, optimizer, lr_scheduler, epoch, args, suffix):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    torch.save(checkpoint, os.path.join(args.results_path, f'{args.experiment_name}.{suffix}.pth'))

def median_frequency_exp(dataset: Dataset, num_classes: int, soft: float):
    # Process the dataset in parallel
    loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)

    # Initialize counts
    classes_freqs = torch.zeros(num_classes, dtype=torch.int64)

    for _, target in loader:
        classes, counts = torch.unique(target, return_counts=True)
        ignore = torch.bitwise_or(classes < 0, classes >= num_classes)
        classes_freqs.index_add_(0, classes[~ignore], counts[~ignore])

    zeros = classes_freqs == 0
    if zeros.sum() != 0:
        print("There are some classes not present in the training samples")

    result = classes_freqs.median() / classes_freqs
    result[zeros] = 0  # avoid inf values
    return result ** soft

def train_epoch(model, dataloader, criterion, optimizer, lr_scheduler, device):
    model.train()
    mean_loss = torchmetrics.MeanMetric().to(device)

    for input, target in dataloader:
        if isinstance(input, list):
            input = [i.to(device) for i in input]
        else:
            input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss.update(loss)

    lr_scheduler.step()
    return mean_loss.compute()

def evaluate(model, dataloader, criterion, num_classes, device):
    model.eval()
    mean_loss = torchmetrics.MeanMetric().to(device)
    class_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task='multiclass', average='none').to(device)

    with torch.inference_mode():
        for input, target in dataloader:
            if isinstance(input, list):
                input = [i.to(device) for i in input]
            else:
                input = input.to(device)
            target = target.to(device)

            output = model(input)

            mean_loss.update(criterion(output, target))
            class_iou.update(output, target)

    mean_loss = mean_loss.compute()
    class_iou = class_iou.compute()
    miou = class_iou[1:].mean()
    iou_std = class_iou[1:].std()
    return mean_loss, class_iou, miou, iou_std

def main(args):
    args.experiment_name = f'{args.model}.{args.input_mode}.{args.target_mode}.{str(uuid.uuid4())[:4]}'
    print(args.experiment_name)

    if ',' in args.input_mode:
        # Use multimodal dataset
        args.input_mode = args.input_mode.split(',')

    train_data = SpectralWasteSegmentation(args.data_path, split='train', input_mode=args.input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTrain(), target_type='')
    val_data = SpectralWasteSegmentation(args.data_path, split='val', input_mode=args.input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTest(), target_type='')
    test_data = SpectralWasteSegmentation(args.data_path, split='test', input_mode=args.input_mode, target_mode=args.target_mode, transforms=SemanticSegmentationTest(), target_type='')

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=30)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=30)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=30)

    model = models.create_model(args.model, train_data.num_channels, train_data.num_classes).to(args.device)
    optimizer, lr_scheduler = models.create_optimizers(args.model, model, args.max_epoch)

    # Calculate loss weights and define loss
    loss_weights = median_frequency_exp(train_data, train_data.num_classes, 0.12)
    criterion = torch.nn.CrossEntropyLoss(loss_weights.to(args.device))

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        if not args.test_only:
            args.start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        # Evaluate model
        test_loss, test_class_iou, test_miou, test_iou_std = evaluate(model, test_dataloader, criterion, train_data.num_classes, args.device)
        print(f'test/loss: {test_loss} | test/class_iou: {test_class_iou.tolist()} | test/miou: {test_miou}, test/iou_std: {test_iou_std}')
        return

    # Start logging
    if args.wandb:
        wandb.init(project=args.wandb, entity='separa', name=args.experiment_name, config=args)

    # Train
    os.makedirs(args.results_path, exist_ok=True)
    best_val_miou = 0

    for epoch in range(args.start_epoch, args.max_epoch):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler, args.device)
        val_loss, val_class_iou, val_miou, val_iou_std = evaluate(model, val_dataloader, criterion, train_data.num_classes, args.device)

        if val_miou > best_val_miou:
            save_checkpoint(model, optimizer, lr_scheduler, epoch, args, 'best')
            best_model = model

        print(f'epoch: {epoch:04d} | train/loss: {train_loss:.4f} | val/loss: {val_loss:.4f} | val/miou: {val_miou:.4f}')

        if args.wandb:
            val_class_iou = {f'val/iou_{train_data.classes_names[i]}': val_class_iou[i] for i in range(train_data.num_classes)}
            wandb.log({
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/miou": val_miou,
                "val/iou_std": val_iou_std,
                **val_class_iou
            })

    save_checkpoint(model, optimizer, lr_scheduler, epoch, args, 'last')

    # Evaluate the best model
    test_loss, test_class_iou, test_miou, test_iou_std = evaluate(best_model, test_dataloader, criterion, train_data.num_classes, args.device)
    print(f'test/loss: {test_loss} | test/miou: {test_miou}')

    if args.wandb:
        test_class_iou = {f'test/best_iou_{train_data.classes_names[i]}': test_class_iou[i] for i in range(train_data.num_classes)}
        wandb.log({
            "test/best_loss": test_loss,
            "test/best_miou": test_miou,
            "test/best_iou_std": test_iou_std,
            **test_class_iou
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--data-path', type=str, default='data/spectralwaste_segmentation')
    parser.add_argument('--results-path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda')
    # setting
    parser.add_argument('--model', type=str, default='mininet')
    parser.add_argument('--input-mode', type=str, default='rgb')
    parser.add_argument('--target-mode', type=str, default='labels_rgb')
    # training
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--resume', type=str, default='', help='Path of a checkpoint')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--wandb', type=str, default='', help='W&B project name')

    args = parser.parse_args()
    main(args)