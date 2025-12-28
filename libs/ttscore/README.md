# ttscore

TTS (Text-to-Speech) 抽象层，支持多种 TTS 提供者的可插拔接入。

## 功能

- **Provider 抽象**：统一的 TTS 接口，方便切换不同后端
- **多提供者支持**：
  - Edge TTS (免费，微软 Edge 浏览器 TTS)
  - Azure TTS (付费，高质量)
  - CosyVoice (本地部署)

## 使用

```python
from libs.ttscore import get_tts_provider

# 使用 Edge TTS (默认)
provider = get_tts_provider("edge")
output_path = await provider.synthesize(
    text="你好，欢迎收听本期小说",
    output_path=Path("output.mp3"),
    voice="zh-CN-XiaoxiaoNeural",
)

# 列出可用音色
voices = await provider.list_voices()
```

## 安装依赖

```bash
# Edge TTS
uv pip install edge-tts

# Azure TTS
uv pip install azure-cognitiveservices-speech
```

## 添加新 Provider

1. 在 `providers/` 下创建新文件
2. 继承 `TTSProvider` 基类
3. 实现 `synthesize()` 和 `list_voices()` 方法
4. 在 `service.py` 中注册
