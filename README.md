# Text Embedding
![Authour](https://img.shields.io/badge/Author-stanleylsx-red.svg) 
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)
[![torch_version](https://img.shields.io/badge/torch-2.0%2B-pink.svg)](requirements.txt)  

## Introduction
本项目用于训练基于双塔模型的给LLM召回相关阅读理解文本和进行句子相似度计算的框架


## Updates
Date| Detail
:---|---
2023-11-21|增加Bert的层次位置编码
2023-11-07|增加fp16混合精度训练
2023-11-03|增加[mteb](https://github.com/embeddings-benchmark/mteb)评测
2023-11-01|加入[ewc loss](https://arxiv.org/abs/1612.00796)
2023-10-27|初始仓库

## Requirement
几个重要环境：
* python：3.10+  
* torch：2.0.1+  
其它环境见requirements.txt  

## Feature

### Supported Models
能够支持的一些模型基座：  

Base Model|link
:---------|--------
XLMRoberta|[e5](https://huggingface.co/intfloat/multilingual-e5-base)
Bert      |[bge](https://huggingface.co/BAAI/bge-base-zh-v1.5)、[piccolo](https://huggingface.co/sensenova/piccolo-base-zh)、[simbert](https://huggingface.co/WangZeJun/simbert-base-chinese)、[m3e](https://huggingface.co/moka-ai/m3e-base)、[gte](https://huggingface.co/thenlper/gte-base-zh)

请从config.py文件中的configure里面修改使用的模型和获取embedding的方式：
```
configure = {
    # 模型类别，支持Bert和XLMRoberta
    'model_type': 'Bert',
    # 获取Embedding的方法，支持cls、last-avg
    'emb_type': 'last-avg',
}
```

### Train Method
Method            |Supported| 
:-----------------|---------|
Cosent            | ✅     |
SimCSE-supervise  | ✅     |
SimCSE-unsupervise| ✅     |

请从config.py文件中的configure里面修改训练的方式：
```
configure = {
    # 训练方式
    # 支持的训练方式有cosent、simcse_sup、simcse_unsup
    'train_type': 'cosent'
}
```

## Mode  
项目提供了四种模式，如下： 

Mode              |Detail                           | 
:-----------------|---------------------------------|
train             | 训练相似度模型                   |
get_embedding     | 获取句子的Embedding              |
predict_one       | 在main.py中写两个句子进行预测测试 |
convert_onnx      | 将torch模型保存onnx文件以便于部署 |
mteb              | 支持通过mteb跑通用评估集         |

## Getting start
项目只需要在config.py中配置好所有策略，然后点击main.py即可运行，没有其他的入口。  
### Train  
**【step1】** 在config.py中，如果使用Cosent训练则train_type选择cosent，使用SimCSE-supervise则train_type选择simcse_sup，使用SimCSE-unsupervise则train_type选择simcse_unsup；
```
# 训练方式
# 支持的训练方式有cosent、simcse_sup、simcse_unsup
'train_type': 'cosent',
```
**【step2】** 在config.py中，准备好数据并且处理格式如/datasets/中的数据格式，如consent的格式看datasets/cosent/train.csv，然后分割训练集、验证集，在config.py文件中配置好训练集、验证集的地址；
```
# 训练数据集
'train_file': 'datasets/cosent/train.csv',
# 验证数据集，必须是pairdata
'val_file': 'datasets/cosent/val.csv',
# 测试数据集
'test_file': 'datasets/cosent/test.csv',
```
**【step3】** 
在config.py文件中配置好模型保存的地址和从huggingface上得到的模型tag，这里使用了商汤的[piccolo-base-zh](https://huggingface.co/sensenova/piccolo-base-zh)；
```
# 模型保存的文件夹
'checkpoints_dir': 'checkpoints/piccolo/first_version',
# 模型的名字
'model_name': 'debug.bin',
# 预训练模型细分类（直接填huggingface上的模型tag）
'hf_tag': 'sensenova/piccolo-base-zh',
```
**【step4】**
如果没有其它的需求，在config.py中修改mode为train，然后点击main.py即可运行训练；
```
# 模式
mode = 'train'
```

***注(1):训练支持使用苏神的层次分解位置编码扩充Bert的长度，也支持EWC方法进行训练，支持FP16训练，都可以通过config.py中的相关key进行开启。***

## Citation

如果你在研究中使用了该项目，请按如下格式引用：

```latex
@misc{Text Embedding,
  title={Text Embedding: A tool for training text representations.},
  author={Shouxian Li},
  year={2023},
  howpublished={\url{https://github.com/stanleylsx/text_embedding}},
}
```

## Reference
[MTEB embedding排行榜](https://huggingface.co/spaces/mteb/leaderboard)  
[CoSENT方法](https://kexue.fm/archives/8847)  
[SimCSE损失函数](https://github.com/yangjianxin1/SimCSE)  
[层次分解位置编码，让BERT可以处理超长文本](https://www.spaces.ac.cn/archives/7947)
[Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796)