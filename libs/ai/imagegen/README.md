# imagegen

图像生成抽象层，支持多种生图后端的可插拔接入。

## 功能

- **Provider 抽象**：统一的图像生成接口，方便切换不同后端
- **多提供者支持**：
  - SD WebUI API (Stable Diffusion WebUI)
  - ComfyUI (节点式工作流)
  - 本地模型加载 (FLUX, SD3)

## 使用

```python
from imagegen import get_imagegen_provider
# 使用 SD WebUI
provider = get_imagegen_provider(
    "sd_webui",
    base_url="http://localhost:7860",
)

output_path = await provider.generate(
    prompt="一个美丽的森林，阳光透过树叶",
    output_path=Path("output.png"),
    width=1280,
    height=720,
    negative_prompt="低质量, 模糊",
)

# 列出可用模型
models = await provider.list_models()
```

## 配置

通过环境变量配置默认后端：

```bash
IMAGEGEN_PROVIDER=sd_webui
IMAGEGEN_SD_WEBUI_URL=http://localhost:7860
```

## 添加新 Provider

1. 在 `providers/` 下创建新文件
2. 继承 `ImageGenProvider` 基类
3. 实现 `generate()` 和 `list_models()` 方法
4. 在 `service.py` 中注册
