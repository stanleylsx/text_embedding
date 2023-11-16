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
