# 南邮公开资料离线问答

面向《大数据项目实践》的本地系统：采集南京邮电大学公开网页与附件，建成可追溯的 SQLite 知识库，用词法检索 + 结构化计算回答问题，最终由 **Qwen3-0.6B + llama.cpp** 组织无计算结果的开放题。正式验收不调用外部模型服务。

正确率优先：表格筛选、汇总、比例和费用由程序计算；小模型不得改写已验证数字。

## 环境

macOS 与 Ubuntu 使用同一套代码和同一条安装命令。Python 3.10 及以上即可。

- 问答与计算读 `data/db/njupt.sqlite`。仓库里是去掉运行痕迹后的压缩快照 `data/db/njupt.sqlite.zst`（Git LFS）。本机已有 sqlite 时不会覆盖。
- 开放题需要 `llama-completion`（llama.cpp）和 `models/Qwen3-0.6B-Q8_0.gguf`。计算题不依赖模型。
- 扫描件 OCR 需要 Tesseract，并带 `chi_sim`。可用环境变量 `ENABLE_OCR=false` 关闭。程序只在 `PATH` 里找 `tesseract` 和 `llama-completion`，也可用 `LLAMA_BIN` 指定可执行文件。

macOS：

```bash
brew install python@3.12 llama.cpp tesseract tesseract-lang zstd git-lfs
```

Ubuntu：

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng zstd git git-lfs
```

Ubuntu 的 `llama-completion` 从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 安装对应的 Ubuntu 包，并把所在目录加入 `PATH`。不要用会把报错打进答案的 `llama-cli`。

两边都执行：

```bash
git lfs install
git clone https://github.com/wilbertyuan/asknjupt.git
cd asknjupt
git lfs pull
bash scripts/setup.sh
source .venv/bin/activate
python -m njuptqa status
```

`setup.sh` 会创建虚拟环境、安装 Python 依赖、拉 LFS 快照、解压数据库，并在模型缺失时下载 GGUF。

## 命令

所有子命令均由 `python -m njuptqa` 提供：

```bash
python -m njuptqa ingest-hosts      # 探测 njupt-suburl.txt，公开站入种子
python -m njuptqa discover          # 登记种子并一跳扩展
python -m njuptqa crawl --max-pages 1500
python -m njuptqa parse
python -m njuptqa validate
python -m njuptqa index
python -m njuptqa answer "2026年综合评价测试费标准是多少？"
python -m njuptqa evaluate --split dev --llm off
python -m njuptqa export --out logs/test_results.csv --llm off
python -m njuptqa integrity
python -m njuptqa update --max-pages 80
python -m njuptqa pipeline --max-pages 220
python -m njuptqa ingest-local        # 把 其他资料/ 纳入同一套来源链
python -m njuptqa status
```

## 断网验收

1. 联网完成 `pipeline`（或已有 `data/`）和 `scripts/download_model.sh`。
2. 断开网络后只运行 `answer` / `evaluate` / `export` / `integrity` / `status`。这些命令读本地 SQLite 与 GGUF，不请求外网。
3. 模型缺失时计算题仍走程序答案。
4. 本仓库未做物理拔线实验；若验收现场断网，按上面步骤冷启动即可。

## 目录

| 路径 | 作用 |
| --- | --- |
| `data/raw/` | 原始字节，按内容哈希存放。体积大，不进仓库 |
| `data/db/njupt.sqlite.zst` | 可直接问答的知识库快照 |
| `data/db/njupt.sqlite` | 解压后的本地库，不进仓库 |
| `data/registries/` | 种子与覆盖矩阵 |
| `models/` | Qwen3-0.6B GGUF 与分词器。权重用脚本下载 |
| `eval/` | 开发集 / 留出集 |
| `logs/` | 采集清单、评测、CSV |
| `assumptions.md` | 未决规则 |
| `PROJECT_STATE.md` | 进度与阻塞 |

## 设计要点

1. 只采集南京邮电大学及部门、学院公开页，以及这些页面实际链出的官方文件。不登录、不绕过认证。来源必须来自官网目录或页面真实链接。
2. HTML 表格展开 rowspan/colspan；PDF 用 pypdf 抽文本。
3. 招生计划、收费规则、实验中心、师资指标进入领域表，每条回指文档与定位。
4. 检索默认 FTS5 + 汉字 bigram，embedding 关闭。
5. 选科“且/或/不限”、学费上浮、折合反推、表内占比由受限算子执行。
6. 输出校验：计算结果中的数字必须出现在最终答案，否则回退程序表述。

## 已知限制

- 需登录办事系统不采集。扫描件在文字层不足时用 Tesseract OCR，可用 `ENABLE_OCR=false` 关闭。
- 本科培养方案已覆盖计算机、管理、自动化、理学院等公开 PDF，研究生培养方案补入了 2020–2026 级多份原文，仍不是全校每个本科专业都有。就业质量报告已入库 2018–2022 届；其中 2019 年终就业率 97.84%、2020 年终总就业率 95.33%。2018 届正文只给出分学历就业率，未再合成全校一个数。2021、2022 届落实率主要在图里，未把图中相邻百分数写成事实。`zsjyc.njupt.edu.cn` 仍无法解析；2016、2017 届和 2023 届及以后的报告在当前公开栏目里没有文件。
- 不宣称已收集南邮全部公开信息。
