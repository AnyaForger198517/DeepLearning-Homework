import matplotlib.pyplot as plt
import os
import numpy as np

epoch = 50

def plot_line(args, y, path, title):
    plt.figure(figsize=(8, 8))
    plt.plot(range(y.shape[0]), y, color='r', linestyle='-', marker='.')
    plt.xlabel('Epoch')
    plt.ylabel(title)
    plt.savefig(path)


def metrics_plot(metrics_top1, metrics_top5, imgs_path, model_list=None):
    cnt = 0
    color = ['r', 'b', 'orange', 'green', 'purple']
    line = ['-', '-.', '-', '-.']
    plt.figure(figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.title('val-top1')
    plt.ylabel('top1')
    plt.xlabel('epoch')
    for (metric, model_setting) in zip(metrics_top1, model_list):
        plt.plot(list(range(epoch)), metric, c=color[cnt], linestyle=line[cnt],  label=model_setting)
        cnt += 1
    plt.grid()
    plt.legend(fontsize=9)

    plt.subplot(1, 2, 2)
    cnt=0
    plt.title('val-top5')
    plt.ylabel('top5')
    plt.xlabel('epoch')
    for (metric, model_setting) in zip(metrics_top5, model_list):
        plt.plot(list(range(epoch)), metric, c=color[cnt], linestyle=line[cnt],  label=model_setting)
        cnt += 1

    plt.grid()
    plt.legend(fontsize=9)
    plt.savefig(imgs_path)

if __name__ == "__main__":
    root = "./saved_models"
    imgs_root = './saved_res_imgs'
    if not os.path.exists(imgs_root):
        os.makedirs(imgs_root)
    WD0p01_models = [i for i in os.listdir(root)]
    # print(WD0p01_models)
    top1 = []
    top5 = []

    for dir_name in WD0p01_models:
        top1_file = os.path.join(root + '/' + dir_name, 'top1.npy')
        top5_file = os.path.join(root + '/' + dir_name, 'top5.npy')
        top1.append(np.load(top1_file)[:50])
        top5.append(np.load(top5_file)[:50])

    img = os.path.join(imgs_root, 'metric_of_basic.png')
    metrics_plot(top1, top5, img, model_list=WD0p01_models)
