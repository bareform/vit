from models import (
    ViT
)

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import torchutils

from .hf_utils import (
    ImageDatasetWithLabels
)

def parse_args():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pretrained training configuration.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet-1k"],
        help="Dataset to train on. Must be one of: `imagenet-1k`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads to use in data loading (default: 4)"
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=True,
        help="Whether to pin memory for data loading (default: True)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1500,
        help="Number of epochs to train for (default: 1500).",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Height of the input images (default: 256).",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Width of the input images (default: 256).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=48,
        help="Hidden dimension (default: 48).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size (default: 16).",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=4,
        help="Number of transformer blocks (default: 4).",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4).",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        default=32,
        help="Dimension of each attention head (default: 32).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate (default: 0.0002).",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs for linear learning-rate warmup (default: 10).",
    )
    parser.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.001,
        help="Warmup starting learning rate as a fraction of --lr (default: 0.001).",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.000001,
        help="Minimum learning rate reached by cosine decay (default: 0.000001).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 (default: 0.9).",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 (default: 0.999).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(".", "checkpoints"),
        help="Directory to save checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--save_checkpoint_interval",
        type=int,
        default=1000,
        help="Interval (in epochs) at which to save model checkpoints (default: 1000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Train with torch.compile.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    args = parser.parse_args()    
    return args

def main():
    args = parse_args()
    pad_length = len(str(args.num_epochs))

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"Training on {args.dataset}")
    if args.dataset == "imagenet-1k":
        hf_dataset = load_dataset("ILSVRC/imagenet-1k", split="train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop((args.image_height, args.image_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        dataset = ImageDatasetWithLabels(
            hf_dataset=hf_dataset,
            transform=transform
        )
        def collate_fn(batch):
            images, labels = zip(*batch)
            return torch.stack(images, dim=0), torch.tensor(labels)
        dataloader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate_fn,
        )
        dataset = "imagenet-1k"
        img_size = (3, args.image_height, args.image_width)
        num_classes = 1000

    start_epoch = 0
    total_epochs = args.num_epochs
    warmup_epochs = args.warmup_epochs
    warmup_start_factor = args.warmup_start_factor
    min_lr = args.min_lr
    checkpoint = None

    checkpoints = sorted(
        f for f in os.listdir(args.checkpoint_dir)
            if f.endswith(".pth")
    ) if os.path.isdir(args.checkpoint_dir) else []

    resume_checkpoint = (
        os.path.join(args.checkpoint_dir, checkpoints[-1])
            if args.resume and checkpoints
                else None
    )

    if resume_checkpoint is None:
        model = ViT(
            img_size=img_size,
            patch_size=args.patch_size,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            heads=args.heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
        )
        model = model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    else:
        print(f"Resuming from {resume_checkpoint}")

        checkpoint = torch.load(
            resume_checkpoint,
            map_location=device,
            weights_only=True,
        )

        total_epochs = checkpoint["num_epochs"]
        warmup_epochs = checkpoint["warmup_epochs"]
        warmup_start_factor = checkpoint["warmup_start_factor"]
        min_lr = checkpoint["min_lr"]

        model = ViT(
            img_size=checkpoint["img_size"],
            patch_size=checkpoint["patch_size"],
            num_classes=checkpoint["num_classes"],
            hidden_dim=checkpoint["hidden_dim"],
            num_blocks=checkpoint["num_blocks"],
            heads=checkpoint["heads"],
            head_dim=checkpoint["head_dim"],
            dropout=checkpoint["dropout"],
        )
        model = model.to(device)
        model.load_state_dict(checkpoint["vit"])

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
        )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]

        print(f"Resuming at epoch {start_epoch + 1}/{total_epochs}")

    criterion = nn.CrossEntropyLoss()

    dtype = torchutils.get_torch_dtype(args.mixed_precision)
    use_amp = (device.type == "cuda") and (dtype != torch.float32)
    use_scaler = (device.type == "cuda") and (dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    if checkpoint is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if args.use_torch_compile:
        print("Compiling")
        model = torch.compile(model)

        warmup_images = torch.randn(
            1,
            *img_size,
            device=device,
        )

        warmup_labels = torch.zeros(
            1,
            device=device,
            dtype=torch.long,
        )

        optimizer.zero_grad()

        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=use_amp,
        ):
            warmup_logits = model(warmup_images)
            warmup_loss = criterion(warmup_logits, warmup_labels)

        warmup_loss.backward()
        optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        print("Finished warmup")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, total_epochs):
        running_loss = 0.0

        with tqdm(dataloader, desc="Training", unit="batch") as pbar:
            for idx, (images, labels) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=use_amp,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{learning_rate:.2e}",
                })

        scheduler.step()
        average_loss = running_loss / len(dataloader)

        print(f"Epoch: {epoch + 1}/{total_epochs}")
        print(f"Loss: {average_loss:.5f}, Learning Rate: {learning_rate:.2e}")

        if (epoch + 1) % args.save_checkpoint_interval == 0:
            print("Saving checkpoint")
            state_dict = OrderedDict(
                (k.replace("_orig_mod.", ""), v)
                for k, v in model.state_dict().items()
            )

            checkpoint_file = os.path.join(
                args.checkpoint_dir,
                f"{args.dataset}_checkpoint_{epoch + 1:0{pad_length}d}.pth",
            )

            torch.save({
                "dataset": args.dataset,
                "vit": state_dict,
                "img_size": img_size,
                "patch_size": args.patch_size,
                "num_classes": num_classes,
                "hidden_dim": args.hidden_dim,
                "num_blocks": args.num_blocks,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "dropout": args.dropout,
                "num_epochs": total_epochs,
                "warmup_epochs": warmup_epochs,
                "warmup_start_factor": warmup_start_factor,
                "min_lr": min_lr,
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, checkpoint_file)

            checkpoint_file = os.path.join(args.checkpoint_dir, f"{args.dataset}_{epoch + 1:0{pad_length}d}.pth")
            torch.save({
                "dataset": args.dataset,
                "vit": state_dict,
                "img_size": img_size,
                "patch_size": args.patch_size,
                "num_classes": num_classes,
                "hidden_dim": args.hidden_dim,
                "num_blocks": args.num_blocks,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "dropout": args.dropout,
            }, checkpoint_file)

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        new_key = k.replace("_orig_mod.", "")
        state_dict[new_key] = v

    checkpoint_file = os.path.join(args.checkpoint_dir, f"{args.dataset}_{args.num_epochs}.pth")
    torch.save({
        "dataset": args.dataset,
        "vit": state_dict,
        "img_size": img_size,
        "patch_size": args.patch_size,
        "num_classes": num_classes,
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dropout": args.dropout,
    }, checkpoint_file)

if __name__ == "__main__":
    main()
