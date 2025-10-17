import os
from argparse import ArgumentParser
import numpy as np
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from monai.data import (CacheDataset, ThreadDataLoader, decollate_batch,
                        load_decathlon_datalist)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, FocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (AsDiscrete, Compose, CropForegroundd,
                              EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandAffined, RandCropByPosNegLabeld, SpatialPadd,
                              RandShiftIntensityd, ScaleIntensityRanged,
                              Spacingd, ToTensord, RandFlipd, RandRotate90d,
                              RandGaussianNoised, RandGaussianSmoothd,
                              RandScaleIntensityd, RandAdjustContrastd)
from monai.utils import set_determinism
from model_mvn.mvn import mvnNet
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


class CombinedLoss(nn.Module):
    """Combined loss function for better vessel segmentation"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.dice_ce = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
        self.focal = FocalLoss(alpha=0.25, gamma=2.0, to_onehot_y=True)
        
    def forward(self, pred, target):
        dice_ce_loss = self.dice_ce(pred, target)
        focal_loss = self.focal(pred, target)
        return 0.7 * dice_ce_loss + 0.3 * focal_loss


def run_train(args):
    # set random seed
    set_determinism(seed=args['seed'])

    # Using date and hour
    current_date_hour = datetime.datetime.now().strftime("%Y%m%d-%H")
    log_dir = os.path.join("runs", "training_" + current_date_hour)
    writer = SummaryWriter(log_dir)

    # use device for train
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(torch.cuda.is_available())

    # load dataset for train and valid
    datasets = args['dataset']
    train_dataset_list = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key='training')
    valid_dataset_list = load_decathlon_datalist(datasets, is_segmentation=True, data_list_key='validation')

    # Enhanced data augmentation for vessel segmentation
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=(0.3542, 0.3542, 0.3542),
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # More aggressive intensity normalization for vessels
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=800,  # Reduced max for better vessel contrast
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Larger patch size for better vessel continuity
        SpatialPadd(keys=['image', 'label'], spatial_size=(96, 96, 96)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=2,  # Increased positive samples
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # Enhanced augmentation pipeline
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandAffined(keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=0.8,  # Increased probability
                    spatial_size=(96, 96, 96),
                    rotate_range=(0, 0, np.pi / 20),  # Reduced rotation for vessels
                    scale_range=(0.1, 0.1, 0.1),
                    translate_range=(5, 5, 5)),
        # Intensity augmentations for better generalization
        RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.6),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
        RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.25, 1.0), 
                           sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)),
        ToTensord(keys=["image", "label"]),
    ])

    # Validation transforms with larger patch size
    valid_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=(0.3542, 0.3542, 0.3542),
                 mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=800,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ])

    # Enhanced caching and data loading
    train_dataset = CacheDataset(train_dataset_list, transform=train_transforms, 
                                cache_num=50, cache_rate=1.0,  # Increased cache
                                num_workers=args['num_workers'])
    valid_dataset = CacheDataset(valid_dataset_list, transform=valid_transforms, 
                                cache_num=10, cache_rate=1.0,
                                num_workers=args['num_workers'])

    train_loader = ThreadDataLoader(train_dataset, num_workers=0, batch_size=args['batch_size'], shuffle=True)
    valid_loader = ThreadDataLoader(valid_dataset, num_workers=0, batch_size=1, shuffle=False)

    # Larger patch size for better vessel connectivity
    patch_size = (96, 96, 96)

    # Initialize model with enhanced configuration
    model = mvnNet(
        in_chans=1,
        out_chans=args['num_classes'],
        feature_dims=[48, 96, 192, 384, 768],  # Consider [64, 128, 256, 512, 1024] for more capacity
    ).to(device)

    # Load pretrained weights if available
    if os.path.exists(args['pretrain_weights']) and args['use_pretrain']:
        logger.info(f"Loading pretrained weights from {args['pretrain_weights']}")
        checkpoint = torch.load(args['pretrain_weights'], map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    # Enhanced loss function
    loss_function = CombinedLoss(num_classes=args['num_classes'])

    # Improved optimizer with weight decay scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'], 
                                 weight_decay=1e-4, betas=(0.9, 0.999))

    # Enhanced learning rate scheduling
    max_iterations = args['max_iter']
    valid_iter = args['valid_iter']
    
    # Warm-up scheduler
    warmup_iter = max_iterations // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args['lr'],
        total_steps=max_iterations,
        pct_start=0.1,  # 10% warm-up
        anneal_strategy='cos'
    )

    post_label = AsDiscrete(to_onehot=args['num_classes'])
    post_pred = AsDiscrete(argmax=True, to_onehot=args['num_classes'])
    dice_metric = DiceMetric(include_background=False, reduction='mean', get_not_nans=False)
    
    global_step = args['epoch']
    dice_valid_best = 0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    # Mixed precision training for better performance
    scaler = torch.cuda.amp.GradScaler()

    # Early stopping parameters
    patience = 10
    no_improve_count = 0

    logger.info("Starting training with enhanced configuration...")
    
    # start training....
    while global_step < max_iterations:
        model.train()
        epoch_loss = 0
        step = 0
        
        for step, batch in enumerate(train_loader):
            step += 1
            x, y = batch['image'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            scheduler.step()

            epoch_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

            logger.info(
                "Epoch:[{:0>5d}/{:0>5d}]\t train_loss = {:.5f}\t lr = {:.2e}".format(
                    global_step, max_iterations, loss.item(), optimizer.param_groups[0]['lr']))

            # Enhanced validation with Test Time Augmentation (TTA)
            if (global_step % valid_iter == 0 and global_step != 0) or (global_step == max_iterations):
                model.eval()
                dice_values = []
                
                with torch.no_grad():
                    for val_step, batch in enumerate(valid_loader):
                        valid_in, target = batch['image'].to(device), batch['label'].to(device)

                        # Test Time Augmentation - multiple predictions with different overlaps
                        pred1 = sliding_window_inference(valid_in, patch_size, 4, model, overlap=0.5)
                        pred2 = sliding_window_inference(valid_in, patch_size, 4, model, overlap=0.6)
                        pred3 = sliding_window_inference(valid_in, patch_size, 4, model, overlap=0.7)
                        
                        # Ensemble prediction
                        valid_out = (pred1 + pred2 + pred3) / 3.0

                        valid_labels_list = decollate_batch(target)
                        valid_labels_convert = [post_label(valid_label_tensor) for valid_label_tensor in valid_labels_list]

                        valid_output_list = decollate_batch(valid_out)
                        valid_output_convert = [post_pred(valid_pred_tensor) for valid_pred_tensor in valid_output_list]
                        
                        dice_metric(y_pred=valid_output_convert, y=valid_labels_convert)

                mean_dice_val = dice_metric.aggregate().item()
                writer.add_scalar('Metric/Validation_Dice', mean_dice_val, global_step)
                dice_metric.reset()

                epoch_loss /= step
                writer.add_scalar('Loss/Epoch', epoch_loss, global_step)
                epoch_loss_values.append(epoch_loss)
                metric_values.append(mean_dice_val)

                logger.info("Valid step: {:0>5d} mean dice = {:.8f}".format(global_step, mean_dice_val))

                # Enhanced model saving with more metrics
                if mean_dice_val > dice_valid_best:
                    dice_valid_best = mean_dice_val
                    global_step_best = global_step
                    no_improve_count = 0
                    
                    state_dict = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'dice_metric': dice_valid_best,
                        'epoch': global_step_best,
                        'scaler': scaler.state_dict(),
                    }

                    torch.save(state_dict, os.path.join(args['checkpoint_dir'], 'best_model.ckpt'))
                    logger.info(
                        f"Model saved! Best Dice: {dice_valid_best:.6f} at iteration {global_step_best}")
                else:
                    no_improve_count += 1
                    logger.info(
                        f"No improvement. Best Dice: {dice_valid_best:.6f}, Current: {mean_dice_val:.6f}")

                # Early stopping
                if no_improve_count >= patience:
                    logger.info(f"Early stopping after {patience} iterations without improvement")
                    break

            global_step += 1

    writer.close()
    logger.info(f"Training completed. Best Dice Score: {dice_valid_best:.6f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Enhanced 3D vessel segmentation")
    parser.add_argument("--dim_in", type=int, default=1, help="input dimension")
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--max_iter", type=int, default=8000, help="maximum number of iterations")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--num_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--valid_iter", type=int, default=400, help="validation frequency")
    parser.add_argument("--dataset", type=str, default="dataset.json")
    parser.add_argument("--pretrain_weights", type=str, default="./RESULTS/model_best.pt")
    parser.add_argument("--use_pretrain", action="store_true", help="use pretrained weights")
    parser.add_argument("--checkpoint_dir", type=str, default='./RESULTS')
    args = parser.parse_args()

    # create model saved dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Enhanced logging
    logger_file = f"train_Enhanced_Vessel_Seg_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.log"
    logger.add(logger_file, rotation="100 MB", level="INFO")

    run_train(vars(args))