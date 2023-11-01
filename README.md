# Text Embedding
本项目用于训练基于双塔模型的给LLM召回相关阅读理解文本和进行句子相似度计算的框架


## Updates
Date| Detail
:---|---
2023-10-27|初始仓库

## Requirement
几个重要环境：
* python：3.10+  
* torch：2.0.1+  
其它环境见requirements.txt  

## Feature

### Supported Base Models
能够支持的一些模型基座：  
Base Model|link
:---------|--------
XLMRoberta|[e5](https://huggingface.co/intfloat/multilingual-e5-base)
Bert      |[bge](https://huggingface.co/BAAI/bge-base-zh-v1.5)、[piccolo](https://huggingface.co/sensenova/piccolo-base-zh)、[simbert](https://huggingface.co/WangZeJun/simbert-base-chinese)、[m3e](https://huggingface.co/moka-ai/m3e-base)  

请从config.py文件中的configure里面修改使用的模型：
```
configure = {
    # 支持的有e5、bge、piccolo、simbert、simbert_v2、m3e
    'model_type': 'piccolo',
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

## Reference
[MTEB embedding排行榜](https://huggingface.co/spaces/mteb/leaderboard)  
[CoSENT方法](https://kexue.fm/archives/8847)  
[SimCSE损失函数](https://github.com/yangjianxin1/SimCSE)  
