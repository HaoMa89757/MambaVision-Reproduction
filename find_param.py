import re
from collections import defaultdict

def parse_logs(log_file):
    results = defaultdict(dict)
    current_epoch = 0

    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = re.search(r'Epoch (\d+)/\d+ (started|training completed)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            train_loss_match = re.search(
                r'training completed. Avg Loss: ([\d.]+), LR: ([\d.]+)',
                line
            )
            if train_loss_match:
                results[current_epoch]['train_loss'] = float(train_loss_match.group(1))
                results[current_epoch]['lr'] = float(train_loss_match.group(2))
                continue

            val_match = re.search(
                r'Validation completed - Loss: ([\d.]+), Top1: ([\d.]+)%, Top5: ([\d.]+)%',
                line
            )
            if val_match:
                results[current_epoch]['val_loss'] = float(val_match.group(1))
                results[current_epoch]['top1'] = float(val_match.group(2))
                results[current_epoch]['top5'] = float(val_match.group(3))

    return results


def analyze_results(results):
    analysis = {
        'best_top1': {'value': 0.0, 'epoch': 0},
        'best_top5': {'value': 0.0, 'epoch': 0},
        'min_train_loss': {'value': float('inf'), 'epoch': 0},
        'min_val_loss': {'value': float('inf'), 'epoch': 0},
        'learning_rates': []
    }

    for epoch, metrics in results.items():
        # 更新Top1准确率
        if metrics['top1'] > analysis['best_top1']['value']:
            analysis['best_top1']['value'] = metrics['top1']
            analysis['best_top1']['epoch'] = epoch

        if metrics['top5'] > analysis['best_top5']['value']:
            analysis['best_top5']['value'] = metrics['top5']
            analysis['best_top5']['epoch'] = epoch

        if metrics['train_loss'] < analysis['min_train_loss']['value']:
            analysis['min_train_loss']['value'] = metrics['train_loss']
            analysis['min_train_loss']['epoch'] = epoch

        if metrics['val_loss'] < analysis['min_val_loss']['value']:
            analysis['min_val_loss']['value'] = metrics['val_loss']
            analysis['min_val_loss']['epoch'] = epoch
    return analysis


def print_report(analysis):
    print("=== 最佳准确率 ===")
    print(f"Top1: {analysis['best_top1']['value']:.2f}% @ Epoch {analysis['best_top1']['epoch']}")
    print(f"Top5: {analysis['best_top5']['value']:.2f}% @ Epoch {analysis['best_top5']['epoch']}\n")

    print("=== 最低损失 ===")
    print(f"训练集: {analysis['min_train_loss']['value']:.4f} @ Epoch {analysis['min_train_loss']['epoch']}")
    print(f"验证集: {analysis['min_val_loss']['value']:.4f} @ Epoch {analysis['min_val_loss']['epoch']}\n")


if __name__ == "__main__":
    log_path = "output/training_log.txt"
    results = parse_logs(log_path)
    analysis = analyze_results(results)
    print_report(analysis)