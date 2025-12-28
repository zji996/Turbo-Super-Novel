# novelcore

小说视频项目管理与流水线编排核心库。

## 功能

- **Project 管理**：创建、更新、查询小说视频项目
- **Scene 管理**：场景拆分与状态追踪
- **Pipeline 编排**：协调 TTS、生图、视频生成任务
- **文案解析**：将文案拆分为场景
- **视频合成**：将各场景视频片段合成最终视频

## 数据模型

- `NovelProject`：项目顶层实体
- `NovelScene`：场景（一段文案对应的音频+图片+视频）
- `NovelPipeline`：流水线执行记录

## 使用

```python
from libs.novelcore import parse_text_to_scenes, NovelProject

# 解析文案
scenes = parse_text_to_scenes(text, delimiter="---")

# 创建项目
project = NovelProject(name="My Novel", source_text=text)
```
