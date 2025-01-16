from torch.optim import Adam, AdamW   # 后者使用 weight decay
import torch
import torch.nn as nn
from tqdm import tqdm
import dataloader
import log
import os
import numpy as np
import random
import argparse
from MobileViT import mobilevit_xxs
import utils


def top1_calcu(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    res = np.sum(preds.argmax(axis=1) == labels) / preds.shape[0]
    # print("top1 is ", res)
    return res


def top5_calcu(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # 获取每个预测的前 5 个类别的索引
    top5_indices = preds.argsort(axis=1)[:, -5:]

    # 检查每个标签是否在前 5 个预测中
    comp = np.array([label in top5 for top5, label in zip(top5_indices, labels)])
    res = np.sum(comp)/comp.shape[0]
    # print("top5 is ", res)
    return res

def train(args, logger, current_model_dir):
    model = mobilevit_xxs(skip_c=args.skip_c).to(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # label_smoothing ?
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loader = dataloader.get_data_loaders(args.data_root, args.batch_size, to_train=True)

    loss_all_epoch = []
    iterate_step = 0
    top1_acc_list = []
    top5_acc_list = []

    # 每训练一轮，就测试一次
    for epoch in tqdm(range(args.epoch), desc='Epoch', total=args.epoch):
        loss_per_epoch = []
        model.train()
        for batch_ids, (images, labels) in enumerate(loader):
            print(f'current batch_idx is {batch_ids}')
            iterate_step += 1
            images = images.to(args.device)
            labels = labels.to(args.device)
            output = model(images)
            optimizer.zero_grad()
            # print(labels.shape)
            # print(output.shape)
            loss = criterion(output, labels)  # preds should be the first parameter!
            loss.backward()
            optimizer.step()
            loss_per_epoch.append(loss.item())  # loss.item() 将tensor转换成python标量数值，之后可以直接存储为npy文件

            # when reach the iteration steps, do logging.
            if iterate_step % 1300 == 0:
                logger.info(f"Epoch: {epoch}/{args.epoch}, Iteration step: {iterate_step}, loss: {loss.item()}")

        # record top1 and top5 accuracy after each epoch
        eval_preds, eval_labels = eval(args, model)
        top1_res = top1_calcu(eval_preds, eval_labels)
        top1_acc_list.append(top1_res)
        top5_res = top5_calcu(eval_preds, eval_labels)
        top5_acc_list.append(top5_res)
        logger.info(f'==============Epoch: {epoch}/{args.epoch},top 1 acc: {top1_res}==============')
        logger.info(f'==============Epoch: {epoch}/{args.epoch},top 5 acc: {top5_res}==============')

        loss_all_epoch.append(np.average(loss_per_epoch))

        # save model every 10 epoch
        # if (epoch+1) % 10 == 0:
        model_saved_path = os.path.join(current_model_dir, f'xxs_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), model_saved_path)
        torch.cuda.empty_cache()

    # 保存训练和测试结果，保存到exp_name相应文件夹
    loss_array_path = os.path.join(current_model_dir, f"loss.npy")
    np.save(loss_array_path, np.array(loss_all_epoch))

    top1_array_path = os.path.join(current_model_dir, f"top1.npy")
    np.save(top1_array_path, np.array(top1_acc_list))

    top5_array_path = os.path.join(current_model_dir, f"top5.npy")
    np.save(top5_array_path, np.array(top5_acc_list))

    return loss_all_epoch, top1_acc_list, top5_acc_list


def eval(args, model):
    # 使用列表收集预测和标签
    preds_list = []
    labels_list = []

    test_loader = dataloader.get_data_loaders(args.data_root, args.batch_size, to_train=False)
    model.eval()

    with torch.no_grad():  # 注意这里需要加上括号
        for _, (images, labels) in enumerate(test_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            outputs = model(images)

            # 收集预测和标签
            preds_list.append(outputs)
            labels_list.append(labels)

    # 最后将列表合并为张量  torch.cat可以将list中的tensor合并
    eval_preds = torch.cat(preds_list).to(args.device)  # 确保在 GPU 上
    eval_labels = torch.cat(labels_list).to(args.device)  # 确保在 GPU 上
    print("eval_preds is \n", eval_preds)
    print("eval_labels is \n", eval_labels)
    return eval_preds, eval_labels  # 返回所有预测结果和真实标签



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        default=8,
                        type=int,
                        help='batch size during training, which is needed.')

    parser.add_argument('--data_root',
                        default='./tiny-imagenet-200',
                        type=str,
                        help='dir store the training and test images.')

    parser.add_argument('--skip_c',
                        action='store_true',
                        help='whether or not concat the input of MobileViT Block, just the red line in figure1-b.')

    parser.add_argument('--epoch',
                        default=100,
                        type=int,
                        help='The max epoch during training.')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='whether or not to use cuda.')

    parser.add_argument('--label_smoothing',
                        type=float,
                        default=0.0,
                        help='the label smoothing rate used in classification.')

    parser.add_argument('--image_size',
                        default=256,
                        type=int,
                        help='Image size actually used in training and test, all input image will be resized.')

    parser.add_argument('--learning_rate',
                        default=1e-5,
                        type=float,
                        help='The initial learning rate for Adam.')

    parser.add_argument('--exp_name',
                        type=str,
                        required=True,
                        help='dir which stores current training res.')

    parser.add_argument('--saved_models_dir',
                        default="./saved_models",
                        type=str,
                        help='the dir which stores the training res models.')

    parser.add_argument('--eval_batch_size',
                        type=int,
                        default=16,
                        help="batch size during eval.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed for initialization')

    parser.add_argument('--saved_img_dir',
                        type=str,
                        default='./img',
                        help="whether or not to train a model.")

    args = parser.parse_args()
    log_file = "./train.log"
    logger = log.create_logger(log_file)

    # 确保结果可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 实验还要使用cuda，还需要设置cuda种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.saved_img_dir):
        os.makedirs(args.saved_img_dir)

    # 如果在未训练之前，就存在本次exp_name的模型文件夹，停止训练！
    current_model_dir = os.path.join(args.saved_models_dir, args.exp_name)
    if os.path.exists(current_model_dir):
        raise FileExistsError(f"The directory {args.exp_name} should not exist in {args.saved_models_dir}.")
    if not os.path.exists(current_model_dir):
        os.makedirs(current_model_dir)

    # 开始记录
    logger.info("Start training...")
    loss_res, top1_res, top5_res = train(args, logger, current_model_dir)
    logger.info("Stop training...")

    path_loss = os.path.join(args.saved_img_dir, 'loss_res.png')
    path_top1 = os.path.join(args.saved_img_dir, 'top1_res.png')
    path_top5 = os.path.join(args.saved_img_dir, 'top5_res.png')
    utils.plot_line(args, loss_res, path_loss, title='loss')
    utils.plot_line(args, top1_res, path_top1, title='top1')
    utils.plot_line(args, top5_res, path_top5, title='top5')


if __name__ == "__main__":
    main()
