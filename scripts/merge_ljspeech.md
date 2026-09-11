# LJSpeech 连续短音频合并

依赖：`pip install numpy soundfile`。不需要 GPU 或模型。

输入为 UTF-8（支持 BOM）清单：

```text
成音频文件夹/切片_1.wav|挺全面的哈，
成音频文件夹/切片_2.wav|接下来我们继续。
```

路径兼容正反斜杠。相对路径默认相对清单同目录的 `wavs` 文件夹，例如 `metadata.csv` 中的 `切片_1.wav` 会解析为 `wavs/切片_1.wav`，无需传 `--audio-root`。可用 `--audio-root` 显式覆盖。标准三列 LJSpeech 可用 `--text-column 2` 选择规范化文本。文件扩展名可以是 `.csv` 或 `.txt`，内容分隔符仍为 `|`。

先检查计划（不写文件）：

```powershell
python scripts/merge_ljspeech.py --input "D:/dataset/metadata.txt" --output-dir "D:/dataset/merged_preview" --target-seconds 60 --max-seconds 90 --dry-run
```

生成前 20 条试听：

```powershell
python scripts/merge_ljspeech.py --input "D:/dataset/metadata.txt" --output-dir "D:/dataset/merged_preview" --target-seconds 60 --max-seconds 90 --limit 20
```

确认后换一个新输出目录、去掉 `--limit 20` 处理全量。`--max-clips 4` 限制最多四条一组，并非保证每组四条。

默认保持清单顺序。只有同一输入清单、同一文件夹、文件名末尾数字之前的前缀相同、编号递增 1 且采样率/声道相同，才会拼接。遇到编号缺口或格式变化就另起一组；没有数字编号的文件单独保留。若清单是 1、10、2 这种顺序，可加 `--order natural`，但应先确认编号确实代表时间顺序。

累计达到目标时长就结束；加入下一条会超过最大时长时提前结束。尾部短样本保留，不丢弃。单条已超过最大时长、重复路径、缺失文件、空文本会报错。脚本不做语义分段或声场检测：同前缀连续编号仍可能不是连续录音，需人工抽查。

直接连接波形，保留原始静音；不重采样、不淡化、不插入静音。输出 float32 WAV，避免额外 PCM 量化，但比 PCM16 更占磁盘。采样率和声道保持原样。默认中文文本直接连接，不添加标点；英文可以用 `--text-joiner " "`。

输出：

- `wavs/`：合并音频。
- `metadata.txt`：相对输出目录的 `路径|文本`。若将它再次输入本脚本，需显式设置 `--audio-root` 为该输出目录（路径已带 `wavs/`）。
- `train_raw.jsonl`：含绝对音频路径、文本、语言，可交给项目的 `prepare_data.py` 重新编码，不含参考音频或时长条件。
- `sources.jsonl`：来源文件、文本、拼接位置（采样帧），方便检查接缝。

不会修改源文件，也不会覆盖已有输出目录。中途失败的输出目录保留用于检查；修正问题后使用新目录重跑。`--limit` 仅限制输出条数，仍会检查所有输入文件头。
