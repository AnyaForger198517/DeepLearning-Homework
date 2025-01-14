## 实验环境
python版本为3.7.16  
```bash
pip install -r requirements.txt
```
相关依赖包如下：  
einops==0.8.0  
matplotlib==3.8.2  
numpy==2.2.1  
Pillow==11.1.0  
torch==2.1.2  
torchvision==0.11.3  
tqdm==4.66.4  
## 数据集下载
使用的Tiny-Imagenet-200数据集可以在该[网站](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200)中下载，克隆项目之后将数据集放置在项目下的第一级目录即可（和saved_models同级）
## 运行方式
```bash
bash run.sh
```
其内容如下：  
```bash
python run.py \
    --exp_name you_exp_name \
    --batch_size 32 \
    --epoch 50 \
    --label_smoothing 0.1 \
    --learning_rate 1e-5
```
## 实验结果

<table style="margin: auto;">
  <tr>
    <td>
      <img src="saved_res_imgs/metric_of_basic.png" alt="改变Label Smoothing(LS)和Weight Decay(WD)实验结果" width="300">
      <p>改变Label Smoothing(LS)和Weight Decay(WD)的消融实验结果</p>
    </td>
    <td>
      <img src="saved_res_imgs/metric_of_WD0p01.png" alt="改变Patch Size(PS)和Skip Connection(skipC)实验结果" width="450">
      <p>改变Patch Size(PS)和Skip Connection(skipC)的消融实验结果</p>
    </td>
  </tr>
</table>


