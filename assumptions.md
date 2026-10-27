# 假设与未决规则

基准日期：2026-10-26。日期未知时保持未知，不把抓取日期写成发布日期。

## 生成模型（验收路径，强制）

- 答案生成模型固定为 **Qwen3-0.6B**，推理引擎为 **llama.cpp** 的 `llama-completion`。macOS 用 Homebrew，Ubuntu 用官方 Release 包。`llama-cli` 会把报错和 logo 打进 stdout，所以验收路径只用 `llama-completion`。可用 `LLAMA_BIN` 指定路径。
- 模型文件：`models/Qwen3-0.6B-Q8_0.gguf`，来源官方 `Qwen/Qwen3-0.6B-GGUF`，量化 **Q8_0**。
- 选择 Q8_0 而不是更激进量化：本机 16GB，0.6B Q8 约 639MB，精度损失小于 Q4，且仍远低于 8GB 试运行目标。
- 验收推理关闭思考模式（`--reasoning off`）。0.6B 思考会占用上下文、拉长输出，且 facilmente 编造数字；确定性计算由程序完成。
- 采样：温度 0.1、top-k 20、top-p 0.9、max tokens 512、ctx 4096。这些是经验起点，以验证集为准，不把“温度为零”当作正确率保证。

## 辅助模型（默认可关闭）

任务书只指定生成模型为 Qwen3-0.6B，**不能**据此认为 embedding / reranker / OCR / 合成标注模型必然允许进入验收路径。

当前实现：

| 组件 | 默认 | 说明 |
| --- | --- | --- |
| 词法检索（FTS5 + 汉字 bigram） | 开 | 不依赖神经网络 |
| 结构化 SQL / 确定性计算 | 开 | 不依赖神经网络 |
| 本地 embedding / reranker | 关 | `config.ENABLE_EMBEDDING=false`，未下载权重 |
| OCR | 开（可关） | 本机 Tesseract `chi_sim+eng`。仅当 PDF 原生文字过少或目标为图片附件时启用。结果写入 `data/processed/ocr/{hash}.json|.txt`，块标记 `ocr_*`，低置信度进质量队列。关闭：`ENABLE_OCR=false`，系统仍可回答已有文本资料 |
| 其他大模型标注 | 关 | 未使用外部模型合成训练数据 |

关闭辅助模型后，系统必须仍能回答。embedding 若以后开启，必须与词法基线比较召回，不得直接宣布“更优”。

分词器 `models/qwen3-0.6b-tokenizer/` 来自同一 Qwen3-0.6B，只用于切块 token 计数，不参与答案生成。

## 数据与访问

- 只采集南京邮电大学及部门、学院公开网页和可公开下载文件。
- `其他资料/` 中由项目组放入的公开文件以 `local://` 入库，访问范围 `local_supplement`，原始路径写在 `sources.discovered_from`。扫描件做 OCR 后进入检索；OCR 人名不稳定，不把识别出的人名当作精确事实。
- 公开网络搜索只用来发现官方 URL。正文必须抓回原文后再入库。`zsjyc.njupt.edu.cn` 与 `njupt.91job.gov.cn` 在本环境 DNS 失败。学校页面实际指向的 `njupt.91job.org.cn` 可以访问，只抓其公开栏目「就业质量年度报告」中的原文 PDF，不把招聘启事或接口摘要写成事实。
- 区分“网络访问受限但内容可公开使用”与“内容需身份授权”。不绕过 CAS / VPN / 登录。
- 子域名以用户提供的 `njupt-suburl.txt` 为准并先行探测，不再猜测新主机。登录/VPN/邮件/支付/个人查询系统只登记不采集。
- 不可达主机记入 `subdomains.json`，不写成“学校没有该站点”。
- 不采集个人成绩、身份信息、私人通信。
- 近重复只聚类审核，不自动删除仅数字、年份或条件不同的版本。
- 生成的问题或改写不得升级为原始事实证据。

## 计算与拒答

- 小模型不执行任意代码。只允许受约束的算子：lookup / filter / join / sum / count / min / max / sort / ratio / difference / percentage / solve_linear。
- 有确定性计算结果时，以程序结果为准；模型只负责组织语句。数字校验失败则回退程序答案。
- 资料不足、解析不可靠、范围歧义、来源冲突时说明限制，不编造精确数字。
- 未指定年份时，优先使用问题主题下最新且仍适用的官方文件，并在答案中写明时间口径。

## 评测

- 开发集与留出集按文档族隔离，不获取未公开测试集。
- 数字、集合、条件优先确定性评分。
- 不报告未经实际运行支持的准确率。
