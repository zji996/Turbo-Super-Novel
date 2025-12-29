"""Text parsing utilities for novel video projects.

Parses raw text into scenes based on delimiter markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SceneData:
    """Parsed scene data from text."""

    sequence: int
    text: str
    image_prompt: str | None = None


def parse_text_to_scenes(
    text: str,
    delimiter: str = "---",
    auto_generate_image_prompt: bool = False,
) -> list[SceneData]:
    """Parse text into scenes based on delimiter.

    Args:
        text: The full text content to parse.
        delimiter: The delimiter that separates scenes.
                   Default is "---" (three dashes on its own line).
        auto_generate_image_prompt: If True, generate a simple image prompt
                                    from the scene text (first sentence).

    Returns:
        List of SceneData objects, one per scene.

    Example:
        >>> text = '''
        ... 第一场景的描述。这是一个美丽的早晨。
        ... ---
        ... 第二场景开始了。主人公走进了森林。
        ... ---
        ... 最终场景。故事结束。
        ... '''
        >>> scenes = parse_text_to_scenes(text)
        >>> len(scenes)
        3
    """
    if not text or not text.strip():
        return []

    # Split by delimiter (on its own line, with optional whitespace)
    pattern = rf"\n\s*{re.escape(delimiter)}\s*\n"
    parts = re.split(pattern, text.strip())

    scenes: list[SceneData] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        image_prompt: str | None = None
        if auto_generate_image_prompt:
            image_prompt = _extract_image_prompt(part)

        scenes.append(
            SceneData(
                sequence=i,
                text=part,
                image_prompt=image_prompt,
            )
        )

    return scenes


def _extract_image_prompt(text: str) -> str:
    """Extract a simple image prompt from scene text.

    Currently uses the first sentence as the prompt.
    Can be enhanced with LLM integration later.
    """
    # Find first sentence (ending with Chinese or English punctuation)
    match = re.search(r"^(.+?[。！？.!?])", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: use first 100 characters
    return text[:100].strip()


def validate_scenes(scenes: list[SceneData]) -> list[str]:
    """Validate parsed scenes and return list of warnings.

    Args:
        scenes: List of parsed scenes.

    Returns:
        List of warning messages (empty if all valid).
    """
    warnings: list[str] = []

    if not scenes:
        warnings.append("No scenes found after parsing")
        return warnings

    for scene in scenes:
        if len(scene.text) < 10:
            warnings.append(
                f"Scene {scene.sequence}: Text too short ({len(scene.text)} chars)"
            )

        if len(scene.text) > 5000:
            warnings.append(
                f"Scene {scene.sequence}: Text very long ({len(scene.text)} chars), may need splitting"
            )

    return warnings
