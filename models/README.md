# 模型

验收生成模型必须是 **Qwen3-0.6B**，文件名：

`models/Qwen3-0.6B-Q8_0.gguf`

来源：官方 `Qwen/Qwen3-0.6B-GGUF`，量化 Q8_0。本机构 16GB，不使用更激进量化。

```bash
bash scripts/download_model.sh
```

分词器（只用于切块计数，不参与生成）放在 `models/qwen3-0.6b-tokenizer/`。

未放入权重时，`python -m njuptqa answer` 仍可用程序计算路径回答结构化题。
