import re
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs, train_losses, val_losses, top1, top5 = [], [], [], [], []

    with open(log_path, 'r') as f:
        log = f.read()

    pattern = re.compile(
    r'Epoch (\d+)/\d+.*?'
        r'Avg Loss: ([\d.]+).*?'
        r'Val Loss: ([\d.]+).*?'
        r'Top1 Acc: ([\d.]+)%.*?'
        r'Top5 Acc: ([\d.]+)%',
        re.DOTALL
    )

    matches = pattern.findall(log)
    for match in matches:
        print(match)
        epochs.append(int(match[0]))
        train_losses.append(float(match[1]))
        val_losses.append(float(match[2]))
        top1.append(float(match[3]))
        top5.append(float(match[4]))

    return epochs, train_losses, val_losses, top1, top5



def plot_metrics(epochs, train_loss, val_loss, top1, top5):
    plt.figure(figsize=(14, 10))

    # 训练/验证损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 验证损失单独
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_loss, 'g--', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.grid(True)

    # Top1 准确率
    plt.subplot(2, 2, 3)
    plt.plot(epochs, top1, 'm-', marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Top1 Validation Accuracy')
    plt.grid(True)

    # Top5 准确率
    plt.subplot(2, 2, 4)
    plt.plot(epochs, top5, 'c-', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Top5 Validation Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    log_path = "output/training_log.txt"  # 修改为你的日志文件路径
    epochs, train_loss, val_loss, top1, top5 = parse_log(log_path)
    plot_metrics(epochs, train_loss, val_loss, top1, top5)