# 本地RAG Demo

这是一个小型 RAG 项目，用于熟悉基础的环境配置以及大模型应用调用，包含以下基本模块：

- **文档加载**：支持上传 PDF 与常见纯文本文件。
- **切块**：按句子和长度做轻量切块。
- **Embedding 检索**：默认使用 `BAAI/bge-small-zh-v1.5`。
- **Reranker 重排**：默认使用 `BAAI/bge-reranker-base`。
- **本地生成**：默认使用 `Qwen/Qwen2.5-3B-Instruct`。
- **Web UI**：使用 Gradio，支持上传文件、建索引、提问与查看最终上下文。

## 1. 项目结构

```text
rag_demo/
├── app.py
├── config.py
├── requirements.txt
├── requirements-gpu.txt
├── README.md
├── sample_docs/
│   ├── VPN使用说明.txt
│   ├── 报销流程.md
│   ├── 请假制度.txt
│   └── 退款规则.txt
└── rag/
    ├── __init__.py
    ├── chunking.py
    ├── embeddings.py
    ├── generator.py
    ├── loaders.py
    ├── pipeline.py
    ├── prompts.py
    ├── rerank.py
    ├── schemas.py
    └── store.py
```

## 2. 运行前准备

### 设备配置要求

1. Linux系统，Windows上安装请使用WSL
2. 12G显存及以上GPU, CUDA版本13.0
3. 20G及以上的储存空间

### 环境安装

以下命令请在当前项目路径下使用，conda环境默认安装在上一级目录conda文件夹中（推荐方式）
```bash
conda create -p ../conda/rag_demo python=3.11
conda activate ../conda/rag_demo
pip install -r requirements.txt
```

### 模型下载

本项目使用以下模型：
- **Embedding 检索**：默认使用 `BAAI/bge-small-zh-v1.5`。
- **Reranker 重排**：默认使用 `BAAI/bge-reranker-base`。
- **本地生成**：默认使用 `Qwen/Qwen2.5-3B-Instruct`。

运行前请将模型下载至本地，以下命令请在当前项目路径下使用：
```bash
mkdir models
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./models/Qwen2.5-3B-Instruct
modelscope download --model BAAI/bge-small-zh-v1.5 --local_dir ./models/bge-small-zh-v1.5
modelscope download --model BAAI/bge-reranker-base --local_dir ./models/bge-reranker-base
```

### 最后确认

运行前，请确保你的当前目录如下：
```text
.
├── app.py
├── config.py
├── models
│   ├── bge-reranker-base
│   │   ├── config.json
│   │   ├── configuration.json
│   │   ├── model.safetensors
│   │   ├── onnx
│   │   │   └── model.onnx
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   ├── sentencepiece.bpe.model
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   ├── bge-small-zh-v1.5
│   │   ├── 1_Pooling
│   │   │   └── config.json
│   │   ├── config.json
│   │   ├── config_sentence_transformers.json
│   │   ├── configuration.json
│   │   ├── model.safetensors
│   │   ├── modules.json
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   ├── sentence_bert_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   └── Qwen2.5-3B-Instruct
│       ├── config.json
│       ├── configuration.json
│       ├── generation_config.json
│       ├── LICENSE
│       ├── merges.txt
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── model.safetensors.index.json
│       ├── README.md
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── vocab.json
├── rag
│   ├── chunking.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── __init__.py
│   ├── loaders.py
│   ├── pipeline.py
│   ├── prompts.py
│   ├── rerank.py
│   ├── schemas.py
│   └── store.py
├── README.md
├── requirements.txt
└── sample_docs
    ├── 报销流程.md
    ├── 请假制度.txt
    ├── 退款规则.txt
    └── VPN使用说明.txt
```

同时，请确认GPU可用，否则可能无法使用：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
若输出True，则表示可用；若输出False，请检查当前GPU驱动

## 3. 运行方式

```bash
python app.py
```

默认会同时输出日志到控制台与项目根目录下的 `rag_demo.log`，便于定位错误堆栈。也可通过环境变量调整：

```bash
export RAG_LOG_LEVEL=INFO
export RAG_LOG_FILE=./rag_demo.log
```

如果你点击“建立索引”后页面长时间无响应，最常见原因是首次加载模型（会比较慢）或本地模型路径不存在。当前版本会在日志与页面错误信息中明确提示模型路径问题，请优先检查 `rag_demo.log`。

启动后在浏览器打开：

```text
http://127.0.0.1:7860
```

## 4. 如何试跑

可以先上传 `sample_docs/` 下的 4 个样例文件，再试这些问题：

- 出差报销要在多久之内提交？需要准备什么材料？
- 首次使用 VPN 应该怎么申请？
- 请假 2 天由谁审批？
- 软件购买后多久内可以退款？

## 5. 默认模型说明

### Embedding

默认使用：

```text
BAAI/bge-small-zh-v1.5
```

优点：模型小、中文检索友好、适合最小 demo。

### Reranker

默认使用：

```text
BAAI/bge-reranker-base
```

适合作为第二阶段相关性排序器，对少量候选片段进行精排。

### 生成模型

默认使用：

```text
Qwen/Qwen2.5-3B-Instruct
```

这个模型体量适中，中文能力较好，适合用作本地轻量中文生成模型。若显存更紧张，也可以改成：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

你可以通过环境变量覆盖默认模型：

```bash
export RAG_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export RAG_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

Windows PowerShell 示例：

```powershell
$env:RAG_LLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
```

## 6. 代码模块与 RAG 对应关系

- `loaders.py`：读取 PDF / 文本文件
- `chunking.py`：把文档切成 chunk
- `embeddings.py`：用 embedding 模型编码文本
- `store.py`：在内存中做向量检索
- `rerank.py`：对召回结果二次重排
- `prompts.py`：组织提示词
- `generator.py`：调用本地中文大模型生成回答
- `pipeline.py`：把各模块串成完整 RAG 链路
- `app.py`：提供交互式 Web UI

## 7. 当前 demo 的边界

这个项目是教学版和最小工程版，不是生产版。当前故意保持轻量，因此有一些边界：

- 向量库是**内存版**，没有落盘与增量更新。
- 暂不支持 OCR，因此扫描 PDF 可能抽不出文本。
- 没有引入 BM25、hybrid retrieval、metadata filtering。
- 没有多轮对话记忆与会话级 query rewrite。
- 没有答案事实核查与自动评测。

这些内容正好可以作为后续章节继续扩展。
