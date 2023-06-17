<p align="center">
  <img src="assets/basicsr_xpixel_logo.png" height=120>
</p>

## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>

[![LICENSE](https://img.shields.io/github/license/xinntao/basicsr.svg)](https://github.com/xinntao/BasicSR/blob/master/LICENSE.txt)
[![PyPI](https://img.shields.io/pypi/v/basicsr)](https://pypi.org/project/basicsr/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/xinntao/BasicSR.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/xinntao/BasicSR/context:python)
[![python lint](https://github.com/xinntao/BasicSR/actions/workflows/pylint.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/pylint.yml)
[![Publish-pip](https://github.com/xinntao/BasicSR/actions/workflows/publish-pip.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/publish-pip.yml)
[![gitee mirror](https://github.com/xinntao/BasicSR/actions/workflows/gitee-mirror.yml/badge.svg)](https://github.com/xinntao/BasicSR/blob/master/.github/workflows/gitee-mirror.yml)

<!-- [English](README.md) **|** [简体中文](README_CN.md) &emsp; [GitHub](https://github.com/xinntao/BasicSR) **|** [Gitee码云](https://gitee.com/xinntao/BasicSR) -->

:rocket: 我们添加了 [BasicSR-Examples](https://github.com/xinntao/BasicSR-examples), 它提供了使用BasicSR的指南以及模板 (以python package的形式) :rocket:

:loudspeaker: **技术交流QQ群**：**320960100** &emsp; 入群答案：**互帮互助共同进步**

:compass: [入群二维码](#e-mail-%E8%81%94%E7%B3%BB) (QQ、微信)  &emsp;&emsp; [入群指南 (腾讯文档)](https://docs.qq.com/doc/DYXBSUmxOT0xBZ05u)

---

<a href="https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="18" alt="google colab logo"></a> Google Colab: [GitHub Link](colab) **|** [Google Drive Link](https://drive.google.com/drive/folders/1G_qcpvkT5ixmw5XoN6MupkOzcK1km625?usp=sharing) <br>
:m: [模型库](docs/ModelZoo_CN.md): :arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ)
:arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing) <br>
:file_folder: [数据](docs/DatasetPreparation_CN.md): :arrow_double_down: [百度网盘](https://pan.baidu.com/s/1AZDcEAFwwc1OC3KCd7EDnQ) (提取码:basr) :arrow_double_down: [Google Drive](https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing) <br>
:chart_with_upwards_trend: [wandb的训练曲线](https://app.wandb.ai/xintao/basicsr) <br>
:computer: [训练和测试的命令](docs/TrainTest_CN.md) <br>
:zap: [HOWTOs](#zap-howtos)

---

BasicSR (**Basic** **S**uper **R**estoration) 是一个基于 PyTorch 的开源图像视频复原工具箱, 比如 超分辨率, 去噪, 去模糊, 去 JPEG 压缩噪声等.

:triangular_flag_on_post: **新的特性/更新**

- :white_check_mark: Oct 5, 2021. 添加 **ECBSR 训练和测试** 代码: [ECBSR](https://github.com/xindongzhang/ECBSR).
  > ACMMM21: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
- :white_check_mark: Sep 2, 2021. 添加 **SwinIR 训练和测试** 代码: [SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang). 更多内容参见 [HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)
- :white_check_mark: Aug 5, 2021. 添加了NIQE， 它输出和MATLAB一样的结果 (both are 5.7296 for tests/data/baboon.png).
- :white_check_mark: July 31, 2021. Add **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
  > CVPR21: BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond
- **[更多](docs/history_updates.md)**

:sparkles: **使用 BasicSR 的项目**

- [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN): 通用图像复原的实用算法
- [**GFPGAN**](https://github.com/TencentARC/GFPGAN): 真实场景人脸复原的实用算法

如果你的开源项目中使用了`BasicSR`, 欢迎联系我 ([邮件](#e-mail-%E8%81%94%E7%B3%BB)或者开一个issue/pull request)。我会将你的开源项目添加到上面的列表中 :blush:

---

如果 BasicSR 对你有所帮助，欢迎 :star: 这个仓库或推荐给你的朋友。Thanks:blush: <br>
其他推荐的项目:<br>
:arrow_forward: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): 通用图像复原的实用算法<br>
:arrow_forward: [GFPGAN](https://github.com/TencentARC/GFPGAN): 真实场景人脸复原的实用算法<br>
:arrow_forward: [facexlib](https://github.com/xinntao/facexlib): 提供实用的人脸相关功能的集合<br>
:arrow_forward: [HandyView](https://github.com/xinntao/HandyView): 基于PyQt5的 方便的看图比图工具<br>
<sub>([ESRGAN](https://github.com/xinntao/ESRGAN), [EDVR](https://github.com/xinntao/EDVR), [DNI](https://github.com/xinntao/DNI), [SFTGAN](https://github.com/xinntao/SFTGAN))</sub>
<sub>([HandyView](https://gitee.com/xinntao/HandyView), [HandyFigure](https://gitee.com/xinntao/HandyFigure), [HandyCrawler](https://gitee.com/xinntao/HandyCrawler), [HandyWriting](https://gitee.com/xinntao/HandyWriting))</sub>

---

## :zap: HOWTOs

我们提供了简单的流程来快速上手 训练/测试/推理 模型. 这些命令并不能涵盖所有用法, 更多的细节参见下面的部分.

| GAN                  |                                              |                                              |          |                                                |                                                        |
| :------------------- | :------------------------------------------: | :------------------------------------------: | :------- | :--------------------------------------------: | :----------------------------------------------------: |
| StyleGAN2            | [训练](docs/HOWTOs_CN.md#如何训练-StyleGAN2) | [测试](docs/HOWTOs_CN.md#如何测试-StyleGAN2) |          |                                                |                                                        |
| **Face Restoration** |                                              |                                              |          |                                                |                                                        |
| DFDNet               |                      -                       |  [测试](docs/HOWTOs_CN.md#如何测试-DFDNet)   |          |                                                |                                                        |
| **Super Resolution** |                                              |                                              |          |                                                |                                                        |
| ESRGAN               |                    *TODO*                    |                    *TODO*                    | SRGAN    |                     *TODO*                     |                         *TODO*                         |
| EDSR                 |                    *TODO*                    |                    *TODO*                    | SRResNet |                     *TODO*                     |                         *TODO*                         |
| RCAN                 |                    *TODO*                    |                    *TODO*                    | SwinIR   | [Train](docs/HOWTOs.md#how-to-train-swinir-sr) | [Inference](docs/HOWTOs.md#how-to-inference-swinir-sr) |
| EDVR                 |                    *TODO*                    |                    *TODO*                    | DUF      |                       -                        |                         *TODO*                         |
| BasicVSR             |                    *TODO*                    |                    *TODO*                    | TOF      |                       -                        |                         *TODO*                         |
| **Deblurring**       |                                              |                                              |          |                                                |                                                        |
| DeblurGANv2          |                      -                       |                    *TODO*                    |          |                                                |                                                        |
| **Denoise**          |                                              |                                              |          |                                                |                                                        |
| RIDNet               |                      -                       |                    *TODO*                    | CBDNet   |                       -                        |                         *TODO*                         |

## :wrench: 依赖和安装

For detailed instructions refer to [docs/INSTALL.md](docs/INSTALL.md).

## :hourglass_flowing_sand: TODO 清单

参见 [project boards](https://github.com/xinntao/BasicSR/projects).

## :turtle: 数据准备

- 数据准备步骤, 参见 **[DatasetPreparation_CN.md](docs/DatasetPreparation_CN.md)**.
- 目前支持的数据集 (`torch.utils.data.Dataset`类), 参见 [Datasets_CN.md](docs/Datasets_CN.md).

## :computer: 训练和测试

- **训练和测试的命令**, 参见 **[TrainTest_CN.md](docs/TrainTest_CN.md)**.
- **Options/Configs**配置文件的说明, 参见 [Config_CN.md](docs/Config_CN.md).
- **Logging**日志系统的说明, 参见 [Logging_CN.md](docs/Logging_CN.md).

## :european_castle: 模型库和基准

- 目前支持的模型描述, 参见 [Models_CN.md](docs/Models_CN.md).
- **预训练模型和log样例**, 参见 **[ModelZoo_CN.md](docs/ModelZoo_CN.md)**.
- 我们也在 [wandb](https://app.wandb.ai/xintao/basicsr) 上提供了**训练曲线**等:

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="./assets/wandb.jpg" height="280">
</a></p>

## :memo: 代码库的设计和约定

参见 [DesignConvention_CN.md](docs/DesignConvention_CN.md).<br>
下图概括了整体的框架. 每个模块更多的描述参见: <br>
**[Datasets_CN.md](docs/Datasets_CN.md)**&emsp;|&emsp;**[Models_CN.md](docs/Models_CN.md)**&emsp;|&emsp;**[Config_CN.md](docs/Config_CN.md)**&emsp;|&emsp;**[Logging_CN.md](docs/Logging_CN.md)**

![overall_structure](./assets/overall_structure.png)

## :scroll: 许可

本项目使用 Apache 2.0 license.<br>
更多关于**许可**和**致谢**, 请参见 [LICENSE](LICENSE/README.md).

## :earth_asia: 引用

如果 BasicSR 对你有帮助, 请引用BasicSR. <br>
下面是一个 BibTex 引用条目, 它需要 `url` LaTeX package.

``` latex
@misc{basicsr,
  author =       {Xintao Wang and Liangbin Xie and Ke Yu and Kelvin C.K. Chan and Chen Change Loy and Chao Dong},
  title =        {{BasicSR}: Open Source Image and Video Restoration Toolbox},
  howpublished = {\url{https://github.com/XPixelGroup/BasicSR}},
  year =         {2022}
}
```

> Xintao Wang, Liangbin Xie, Ke Yu, Kelvin C.K. Chan, Chen Change Loy and Chao Dong. BasicSR: Open Source Image and Video Restoration Toolbox. <https://github.com/xinntao/BasicSR>, 2022.

## :e-mail: 联系

若有任何问题, 请电邮 `xintao.alpha@gmail.com`, `xintao.wang@outlook.com`.

<br>

- **QQ群**: 扫描左边二维码 或者 搜索QQ群号: 320960100   入群答案：互帮互助共同进步
- **微信群**: 我们的群一已经满500人啦，进群二可以扫描中间的二维码；如果进群遇到问题，也可以添加 Liangbin 的个人微信 (右边二维码)，他会在空闲的时候拉大家入群~

<p align="center">
  <img src="https://user-images.githubusercontent.com/17445847/134879983-6f2d663b-16e7-49f2-97e1-7c53c8a5f71a.jpg"  height="300">  &emsp;
  <img src="https://user-images.githubusercontent.com/52127135/172553058-6cf32e10-2959-42dd-b26a-f802f09343b0.png"  height="300">  &emsp;
  <img src="https://user-images.githubusercontent.com/17445847/139572512-8e192aac-00fa-432b-ac8e-a33026b019df.png"  height="300">
</p>



## HowTO use BasicSR

`BasicSR` 的使用方式如下：

- Git clone 整个 BasicSR 的代码。这样可以看到 BasicSR 完整的代码，然后根据你自己的需求进行修改，

```bash
git clone https://github.com/xinntao/BasicSR.git
pip3 install -r requirements.txt
python setup.py develop
```

### 预备

大部分的深度学习项目，都可以分为以下几个部分：

1. **data**: 定义了训练数据，来喂给模型的训练过程
2. **arch** (architecture): 定义了网络结构 和 forward 的步骤
3. **model**: 定义了在训练中必要的组件（比如 loss） 和 一次完整的训练过程（包括前向传播，反向传播，梯度优化等），还有其他功能，比如 validation等
4. training pipeline: 定义了训练的流程，即把数据 dataloader，模型，validation，保存 checkpoints 等等串联起来

当我们开发一个新的方法时，我们往往在改进: **data**, **arch**, **model**；而很多流程、基础的功能其实是共用的。那么，我们希望可以专注于主要功能的开发，而不要重复造轮子。
因此便有了 BasicSR，它把很多相似的功能都独立出来，我们只要关心 **data**, **arch**, **model** 的开发即可。


下面我们就通过一个简单的例子，来说明如何使用 BasicSR 来搭建你自己的项目。

我们提供了两个样例数据来做展示，
1. [BSDS100](https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/BSDS100.zip) for training
1. [Set5](https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/Set5.zip) for validation

在 BasicSR-example 的根目录运行下面的命令来下载：

```bash
python scripts/data_preparation/prepare_example_data.py
```

样例数据就下载在 `datasets/example` 文件夹中。

### 0: 目的

我们来假设一个超分辨率 (Super-Resolution) 的任务，输入一张低分辨率的图片，输出高分辨率的图片。低分辨率图片包含了 1) cv2 的 bicubic X4 downsampling 和 2) JPEG 压缩 (quality=70)。

为了更好的说明如何使用 arch 和 model，我们想要使用 1) 类似 SRCNN 的网络结构；2) 在训练中同时使用 L1 和 L2 (MSE) loss。

那么，在这个任务中，我们要做的是:

1. 构建自己的 data loader
1. 确定使用的 architecture
1. 构建自己的 model

下面我们分别来说明一下。

### 1: data

这个部分是用来确定喂给模型的数据的。

这个 dataset 的例子在[basicsr/data/example_dataset.py](basicsr/data/example_dataset.py) 中，它完成了:
1. 我们读取 Ground-Truth (GT) 的图像。读取的操作，BasicSR 提供了[FileClient](https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/file_client.py), 可以方便地读取 folder, lmdb 和 meta_info txt 指定的文件。在这个例子中，我们通过读取 folder 来说明，更多的读取模式可以参考 [basicsr/data](https://github.com/xinntao/BasicSR/tree/master/basicsr/data)
1. 合成低分辨率的图像。我们直接可以在 `__getitem__(self, index)` 的函数中实现我们想要的操作，比如降采样和添加 JPEG 压缩。很多基本操作都可以在 [[basicsr/data/degradations]](https://github.com/xinntao/BasicSR/blob/master/basicsr/data/degradations.py), [[basicsr/data/tranforms]](https://github.com/xinntao/BasicSR/blob/master/basicsr/data/transforms.py) 和 [[basicsr/data/data_util]](https://github.com/xinntao/BasicSR/blob/master/basicsr/data/data_util.py) 中找到
1. 转换成 Torch Tensor，返回合适的信息

**注意**：
1. 需要在 `ExampleDataset` 前添加 `@DATASET_REGISTRY.register()`，以便注册好新写的 dataset。这个操作主要用来防止出现同名的 dataset，从而带来潜在的 bug
1. 新写的 dataset 文件要以 `_dataset.py` 结尾，比如 `example_dataset.py`。 这样，程序可以**自动地** import，而不需要手动地 import

在 [option 的 example 配置文件中](options/train/example/example_option.yml)使用新写的 dataset：

```yaml
datasets:
  train:  # training dataset
    name: ExampleBSDS100
    type: ExampleDataset  # the class name

    # ----- the followings are the arguments of ExampleDataset ----- #
    dataroot_gt: datasets/example/BSDS100
    io_backend:
      type: disk

    gt_size: 128
    use_flip: true
    use_rot: true

    # ----- arguments of data loader ----- #
    use_shuffle: true
    num_worker_per_gpu: 3
    batch_size_per_gpu: 16
    dataset_enlarge_ratio: 10
    prefetch_mode: ~

  val:  # validation dataset
    name: ExampleSet5
    type: ExampleDataset
    dataroot_gt: datasets/example/Set5
    io_backend:
      type: disk
```

### 2: arch

Architecture 的例子在 [basicsr/archs/example_arch.py](basicsr/archs/example_arch.py)中。它主要搭建了网络结构。

**注意**：
1. 需要在 `ExampleArch` 前添加 `@ARCH_REGISTRY.register()`，以便注册好新写的 arch。这个操作主要用来防止出现同名的 arch，从而带来潜在的 bug
1. 新写的 arch 文件要以 `_arch.py` 结尾，比如 `example_arch.py`。 这样，程序可以**自动地** import，而不需要手动地 import

在 [option 配置文件中](options/train/example/example_option.yml)使用新写的 arch:

```yaml
# network structures
network_g:
  type: ExampleArch  # the class name

  # ----- the followings are the arguments of ExampleArch ----- #
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  upscale: 4
```

### 3: model

Model 的例子在 [basicsr/models/example_model.py](basicsr/models/example_model.py)和[basicsr/losses/example_loss.py](basicsr/losses/example_loss.py)中。它主要搭建了模型的训练过程。
在这个文件中：
1. 我们从 basicsr 中继承了 `SRModel`。很多模型都有相似的操作，因此可以通过继承 [basicsr/models](https://github.com/xinntao/BasicSR/tree/master/basicsr/models) 中的模型来更方便地实现自己的想法，比如GAN模型，Video模型等
1. 使用了两个 Loss： L1 和 L2 (MSE) loss。注意添加文件 [basicsr/losses/example_loss.py](basicsr/losses/example_loss.py)
1. 其他很多内容，比如 `setup_optimizers`, `validation`, `save`等，都是继承于 `SRModel`

**注意**：
1. 需要在 `ExampleModel` 前添加 `@MODEL_REGISTRY.register()`，以便注册好新写的 model。这个操作主要用来防止出现同名的 model，从而带来潜在的 bug
2. 新写的 model 文件要以 `_model.py` 结尾，比如 `example_model.py`。 这样，程序可以**自动地** import，而不需要手动地 import
3. 需要在 `ExampleModel` 前添加 `LOSS_REGISTRY.register()`，以便注册好新写的 loss。这个操作主要用来防止出现同名的 model，从而带来潜在的 bug

在 [option 配置文件中](options/train/example/example_option.yml)使用新写的 model:

```yaml
# training settings
train:
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [50000]
    gamma: 0.5

  total_iter: 100000
  warmup_iter: -1  # no warm up

  # ----- the followings are the configurations for two losses ----- #
  # losses
  l1_opt:
    type: L1Loss
    loss_weight: 1.0
    reduction: mean

  l2_opt:
    type: MSELoss
    loss_weight: 1.0
    reduction: mean
```

### 4: training pipeline

整个 training pipeline 可以复用 basicsr 里面的 [basicsr/train.py]

```python
import os.path as osp

import archs  # noqa: F401
import data  # noqa: F401
import models  # noqa: F401
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)

```

### 5: debug mode

至此，我们已经完成了我们这个项目的开发，下面可以通过 `debug` 模式来快捷地看看是否有问题:

```bash
python basicsr/train.py -opt options/train/example/example_option.yml --debug
```

只要带上 `--debug` 就进入 debug 模式。在 debug 模式中，程序每个iter都会输出，8个iter后就会进行validation，这样可以很方便地知道程序有没有bug啦~

### 6: normal training

经过debug没有问题后，我们就可以正式训练了。

```bash
python basicsr/train.py -opt options/train/example/example_option.yml
```

如果训练过程意外中断需要 resume, 则使用 `--auto_resume` 可以方便地自动resume：
```bash
python train.py -opt options/example_option.yml --auto_resume
```

至此，使用 `BasicSR` 开发你自己的项目就介绍完了，是不是很方便呀~ :grin:

## As a Template

你可以使用 BasicSR-Examples 作为你项目的模板。下面主要展示一下你可能需要的修改。

1. 设置 *pre-commit* hook
    1. 在文件夹根目录, 运行
    > pre-commit install
1. 修改 `LICENSE` 文件<br>
    本仓库使用 *MIT* 许可, 根据需要可以修改成其他许可

使用 简单模式 的基本不需要修改，使用 安装模式 的可能需要较多修改，参见[这里](https://github.com/xinntao/BasicSR-examples/blob/installation/README_CN.md#As-a-Template)

## :e-mail: 联系

如果你有任何问题，或者想要添加你的项目到列表中，欢迎电邮
 `xintao.wang@outlook.com` or `xintaowang@tencent.com`.
