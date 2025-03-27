import logging
from datetime import datetime
import torch
import argparse
from timm import utils
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.MambaVision import MambaVision
from data_process.ImageDataset import ImageDataset
from timm.utils import ModelEmaV2
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from contextlib import suppress

class Trainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()
        self._prepare_data()
        self._configure_optimization()
        logging.info(f"Model initialized on {self.device}")

    def _prepare_data(self):
        try:
            # 训练集数据增强
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # 验证集数据处理
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            # 创建数据集
            train_dataset = ImageDataset(self.args.data_dir_train, transform=train_transform)
            val_dataset = ImageDataset(self.args.data_dir_val, transform=val_transform)

            # 创建数据加载器
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.workers,
                pin_memory=True,
                persistent_workers=True
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.workers,
                pin_memory=True
            )
            logging.info(f"Data loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def _configure_optimization(self):
        try:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )

            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.args.epochs,
                eta_min=self.args.min_lr
            )

            if self.args.smoothing:
                self.criterion = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss()
            logging.info("Optimization configured: SGD with CosineAnnealingLR")
        except Exception as e:
            logging.error(f"Error configuring optimization: {str(e)}")
            raise

    def _build_model(self):
        try:
            self.model = MambaVision(
                depths=[1, 3, 8, 4],
                num_heads=[2, 4, 8, 16],
                window_size=[8, 8, 14, 7],
                dim=80,
                in_dim=32,
                mlp_ratio=4,
                drop_path_rate=0.2,
                num_classes=self.args.num_classes
            ).to(self.device)

            if self.args.pretrained:
                if not self.args.pretrained_path:
                    raise ValueError("Pretrained path is required when using pretrained weights")

                try:
                    checkpoint = torch.load(self.args.pretrained_path, map_location='cpu')

                    # 处理不同的checkpoint格式
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint

                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                    msg = self.model.load_state_dict(state_dict, strict=False)
                    logging.info(f"Loaded pretrained weights from {self.args.pretrained_path}")
                    logging.info(f"Missing keys: {msg.missing_keys}")
                    logging.info(f"Unexpected keys: {msg.unexpected_keys}")
                except Exception as e:
                    logging.error(f"Error loading pretrained weights: {str(e)}")
                    raise

            if self.args.model_ema:
                self.model_ema = ModelEmaV2(
                    self.model,
                    decay=0.9998,
                    device=self.device
                )
                logging.info("Model EMA initialized")
            else:
                self.model_ema = None
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.amp)
            self.autocast = torch.cuda.amp.autocast if self.args.amp else suppress
            logging.info(f"Model built with AMP {'enabled' if self.args.amp else 'disabled'}")
        except Exception as e:
            logging.error(f"Error building model: {str(e)}")
            raise

    def train_epoch(self, epoch):
        self.model.train()
        losses = utils.AverageMeter()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for images, targets in pbar:
            try:
                images, targets = images.to(self.device), targets.to(self.device)

                with self.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.model_ema:
                    self.model_ema.update(self.model)

                losses.update(loss.item(), images.size(0))
                pbar.set_postfix(loss=losses.avg)
            except Exception as e:
                logging.error(f"Error in training step: {str(e)}")
                raise

        current_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        logging.info(f"Epoch {epoch + 1} training completed. Avg Loss: {losses.avg:.4f}, LR: {current_lr:.6f}")
        return losses.avg

    def validate(self):
        self.model.eval()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                try:
                    images, targets = images.to(self.device), targets.to(self.device)
                    with self.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)

                    acc1, acc5 = utils.accuracy(outputs, targets, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1.item(), images.size(0))
                    top5.update(acc5.item(), images.size(0))
                except Exception as e:
                    logging.error(f"Error in validation step: {str(e)}")
                    raise

        logging.info(f"Validation completed - Loss: {losses.avg:.4f}, Top1: {top1.avg:.2f}%, Top5: {top5.avg:.2f}%")
        return {
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg
        }


def parse_args():
    parser = argparse.ArgumentParser(description='MambaVision Training')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=310, help='训练总轮数')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--min-lr', type=float, default=1e-5, help='最小学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量系数')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--clip-grad', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器进程数')

    # 模型参数
    parser.add_argument('--data-dir-train', type=str, default=r'/root/autodl-tmp/Dataset/train', help='数据集目录')
    parser.add_argument('--data-dir-val', type=str, default=r'/root/autodl-tmp/Dataset/val', help='数据集目录')
    parser.add_argument('--num-classes', type=int, default=100, help='分类类别数')
    parser.add_argument('--pretrained', default=False,)
    parser.add_argument('--pretrained_path', type=str, default=r'')

    # 增强参数
    parser.add_argument('--smoothing', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--ema-decay', type=float, default=0.9998, help='EMA衰减率')
    parser.add_argument('--model-ema', action='store_true', help='启用模型EMA')

    parser.add_argument('--amp', action='store_true', help='启用混合精度训练')

    return parser.parse_args()


def main():
    try:
        args = parse_args()

        # 配置日志系统
        log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

        logging.info("Training started with configuration:")
        for arg, value in vars(args).items():
            logging.info(f"{arg:20}: {value}")

        trainer = Trainer(args)
        best_acc = 0.0

        for epoch in range(args.epochs):
            logging.info(f"Epoch {epoch + 1}/{args.epochs} started")
            train_loss = trainer.train_epoch(epoch)
            eval_metrics = trainer.validate()

            if eval_metrics['top1'] > best_acc:
                best_acc = eval_metrics['top1']
                print(f"New best model saved with Top1 Acc: {best_acc:.2f}%")
                torch.save({
                    'model': trainer.model.state_dict(),
                    'ema': trainer.model_ema.state_dict() if args.model_ema else None,
                    'metrics': eval_metrics
                }, 'best_model1.pth')
                logging.info(f"New best model saved with Top1 Acc: {best_acc:.2f}%")

            logging.info(
                f"Epoch {epoch + 1} Summary: \n"
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {eval_metrics['loss']:.4f}\n"
                f"Top1 Acc: {eval_metrics['top1']:.2f}% | "
                f"Top5 Acc: {eval_metrics['top5']:.2f}%\n"
                f"----------------------------------------"
            )

        logging.info(f"Training completed. Best Top1 Accuracy: {best_acc:.2f}%")
    except Exception as e:
        logging.error(f"Critical error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()