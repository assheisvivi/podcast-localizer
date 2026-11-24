# 英文采访转中文配音：一分钟 MVP

这是一个为国际新闻编辑设计的最小可行版本：英文音频经过转写和中文初译后，程序会强制暂停，等待记者逐段核对，再生成中文配音。

项目介绍页位于 [`docs/index.html`](docs/index.html)，审校工作台位于 [`docs/app.html`](docs/app.html)。工作台需要运行根目录的 FastAPI 服务；仅用 GitHub Pages 发布 `docs/` 时，上传和合成接口不会运行。公开仓库不包含本地课程记录、研究资料、采访或演示音频，以及 `runs/` 中的审校与生成结果。下面的 `trump_victory_speech_65s.mp3` 路径是本地演示样本；从公开仓库克隆后，请换成自己有权使用的音频文件。

## 网页工作台（邀请试用）

网页和 API 由同一个 FastAPI 应用提供。工作台可通过试用码直接打开本机 `input/audio/trump_victory_speech_65s.mp3` 和已有审校草稿，无须消耗转写与翻译额度；该样本仅供本地开发，不应作为公开演示素材。上传新音频时，可编辑本次术语表。工作台按数字、术语、事实提示和引语标记优先核查段落，可筛选、保存审校并导出 `human_edit_log.csv`。只有服务端保存了全部段落的批准状态、且制作人确认发布时披露 AI 声音，才会接受配音请求。

审校会话目前只保存在运行中的服务进程内存中。重启服务或刷新页面后，未导出的工作无法恢复；这仍是单机开发原型，尚无账号、数据库、跨设备保存或生产级限流。

本地启动：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export DEMO_ACCESS_CODE='自己设置的一串长随机试用码'
export OPENAI_API_KEY='有效的 OpenAI API Key'
# 二选一：MiniMax 或 ElevenLabs；OpenAI TTS 可复用上面的 Key
export MINIMAX_API_KEY='有效的 MiniMax API Key'
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

然后打开 `http://127.0.0.1:8000/app.html`，输入试用码并点击“打开本地演示样本”。只体验审校时不需要 OpenAI 或 TTS API Key；生成配音时仍需要相应服务密钥。网页默认使用 MiniMax 配音；如果只配置了 OpenAI Key，在配音服务中改选 OpenAI。`DEMO_ACCESS_CODE` 是访问限制，不能代替正式的用户认证与用量控制。API Key 只放在服务端环境变量，不要提交到 GitHub，也不要填进网页。

如果部署到 Vercel，请从 GitHub 导入**仓库根目录**（包含 `app.py`、`pipeline.py`、`requirements.txt` 和 `docs/`），在项目环境变量中配置 `DEMO_ACCESS_CODE`、`OPENAI_API_KEY` 和所选配音服务的密钥。网页通过同域 `/api/prepare` 与 `/api/synthesize` 调用后端。当前上传上限为 4 MB；录音会传给配置的第三方服务。采访敏感内容应先取得编辑部批准和受访者相关授权，再邀请他人试用。

## 工作流

```text
英文音频
  -> 带时间戳的英文转写
  -> 新闻级中文初译
  -> review.csv（强制人工审校）
  -> 中文 AI 配音
```

当前 V1 只处理单一说话人的约一分钟音频，输出纯中文配音。自动混音、多人分轨和经授权的声音定制属于后续版本。

## 1. 生成转写、初译和审校表

确保环境中已经配置 `OPENAI_API_KEY`，然后运行：

```bash
python3 pipeline.py prepare \
  --input input/audio/trump_victory_speech_65s.mp3 \
  --work-dir runs/trump_demo
```

程序会生成：

- `transcript.json`：英文全文、时间戳和原始转写信息。
- `translation_draft.json`：结构化中文初译。
- `review.csv`：给编辑逐段审校的表格。
- `audit.jsonl`：不包含 API Key 的处理记录。

## 2. 人工审校

用 Excel、Numbers 或任意表格软件打开 `runs/trump_demo/review.csv`：

1. 对照音频检查 `english_transcript`；如有错误，把正确原文写入 `editor_english_transcript`。
2. 检查人名、机构、数字、日期和政治口号。
3. 在 `editor_translation_zh` 写入最终中文；如果 AI 初译无需修改，也应复制过去，留下明确的人工确认痕迹。
4. 在 `editor_notes` 记录事实核查或改动原因。
5. 确认无误后，将该行 `review_status` 改为 `approved`。

检查进度：

```bash
python3 pipeline.py status --review runs/trump_demo/review.csv
```

## 3. 选择高质量中文声音

已经接入两套云端 TTS；原来的 macOS 系统声音只保留为离线兜底：

| 后端 | 适合场景 | 当前默认模型 | 注意事项 |
|---|---|---|---|
| MiniMax | 中文新闻、解说、播客；优先推荐 | `speech-2.8-hd` | 中文控制直接，默认使用平台预置音色，不克隆真人 |
| ElevenLabs | 更强的情绪表现、跨语种项目 | `eleven_v4` | 先在 Voice Library 试听中文，再把 `voice_id` 配进环境变量 |
| Google Chirp 3 HD | 已经使用 Google Cloud、重视企业部署 | 尚未接入 | 普通话可用，价格清晰，后续可作为第三后端 |
| Azure Speech | 机构已有 Azure、需要正式的个人声音同意流程 | 尚未接入 | Personal Voice 要求说话人录制同意声明 |

### 推荐：MiniMax

在 MiniMax 开放平台创建 API Key，然后只在当前终端设置环境变量，不要把密钥写进项目文件：

```bash
export MINIMAX_API_KEY='你的密钥'

python3 pipeline.py synthesize-minimax \
  --review runs/trump_demo_local/review.csv \
  --output runs/trump_demo_local/trump_zh_minimax.mp3
```

默认参数是 `speech-2.8-hd`、预置音色 `male-qn-qingse` 和 `0.96` 倍语速。试听后可替换平台提供的其他音色：

```bash
python3 pipeline.py synthesize-minimax \
  --review runs/trump_demo_local/review.csv \
  --output runs/trump_demo_local/trump_zh_minimax.mp3 \
  --voice '你的 MiniMax voice_id' \
  --speed 0.94
```

### 备选：ElevenLabs

先在 ElevenLabs Voice Library 选择一款普通话试听自然的预置声音，复制它的 Voice ID：

```bash
export ELEVENLABS_API_KEY='你的密钥'
export ELEVENLABS_VOICE_ID='你选中的 voice_id'

python3 pipeline.py synthesize-elevenlabs \
  --review runs/trump_demo_local/review.csv \
  --output runs/trump_demo_local/trump_zh_elevenlabs.mp3
```

也可以用 `--voice-id` 和 `--tts-model` 临时覆盖。默认模型是 `eleven_v4`，中文语言码固定为 `zh`。

## 4. 生成正式中文配音

只有全部分段都标记为 `approved` 后，下面的命令才会运行：

```bash
python3 pipeline.py synthesize-minimax \
  --review runs/trump_demo/review.csv \
  --output runs/trump_demo/trump_zh_approved.mp3
```

为了展示技术链路，可以显式加上 `--allow-unreviewed-demo`。这样得到的文件必须标为“未经人工审校的演示稿”，不得直接发布。

这个开关同样适用于 MiniMax 和 ElevenLabs。例如当前特朗普样本尚未完成人工审校，只能这样生成内部试听：

```bash
python3 pipeline.py synthesize-minimax \
  --review runs/trump_demo_local/review.csv \
  --output runs/trump_demo_local/trump_zh_minimax_unreviewed.mp3 \
  --allow-unreviewed-demo
```

如果暂时没有任何云端 API Key，可把本地转写和翻译草稿导入同一套审校流程，并使用 macOS 的通用普通话声音生成技术演示稿。该声音自然度较低，不建议发布：

```bash
python3 pipeline.py import-draft \
  --transcript runs/trump_demo_local/transcript_verified.json \
  --translation runs/trump_demo_local/translation_draft_local.json \
  --work-dir runs/trump_demo_local

python3 pipeline.py synthesize-local \
  --review runs/trump_demo_local/review.csv \
  --output runs/trump_demo_local/trump_zh_unreviewed_demo.mp3 \
  --allow-unreviewed-demo
```

## 新闻与合规边界

- 默认中文声音是通用 AI 声音，不模仿特朗普或其他真实人物。
- 声音克隆或定制必须取得说话人的明确授权，并保留同意记录。公开人物的公开视频不等于声音克隆授权。
- 对外发布时应清楚告知听众中文声音由 AI 生成。
- 匿名信源、未公开采访和其他敏感音频不应在未获编辑部批准时上传第三方服务；后续版本应提供本地转写和本地翻译选项。
- `review.csv` 是产品的核心，不是临时障碍：最终责任由新闻编辑承担。

## 成本与选型备注

价格会变动，上线前应再次查看厂商官方页面。2026 年 10 月查询时，MiniMax `speech-2.8-hd` 的标价为每百万字符 100 美元、Turbo 为 60 美元；ElevenLabs 的主流多语种/表现力模型页面显示约每千字符 0.10 美元；Google Chirp 3 HD 为每百万字符 30 美元。对一分钟中文稿，实际成本通常很低，选型应优先比较音色、专名发音、数据协议和编辑流程。

- MiniMax API 文档：https://platform.minimax.io/docs/api-reference/speech-t2a-http
- MiniMax 定价：https://platform.minimax.io/subscribe/token-plan?tab=api-enterprise
- ElevenLabs 模型：https://elevenlabs.io/docs/overview/models
- ElevenLabs API 定价：https://elevenlabs.io/pricing/api
- Google Chirp 3 HD：https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd
- Google Cloud TTS 定价：https://cloud.google.com/text-to-speech/pricing
- Azure Personal Voice 同意流程：https://learn.microsoft.com/azure/ai-services/speech-service/personal-voice-create-consent

## 可替换模型

命令行参数可替换转写、翻译、TTS 模型和声音：

```bash
python3 pipeline.py prepare --help
python3 pipeline.py synthesize --help
```

当前默认配置：

- 转写：`whisper-1`，因为 V1 需要分段时间戳。
- 翻译：`gpt-5-mini`，使用严格 JSON Schema 输出。
- 原有配音后端：`gpt-4o-mini-tts`，通用声音 `cedar`。
- 推荐中文配音：MiniMax `speech-2.8-hd`，预置声音 `male-qn-qingse`。
- 高表现力备选：ElevenLabs `eleven_v4`，声音由编辑试听后指定。
