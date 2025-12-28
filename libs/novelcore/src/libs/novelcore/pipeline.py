"""Pipeline orchestration for novel video generation.

Coordinates the execution of TTS, image generation, video generation,
and final composition tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TaskType(str, Enum):
    """Types of tasks in the pipeline."""
    TTS = "TTS"
    IMAGEGEN = "IMAGEGEN"
    VIDEOGEN = "VIDEOGEN"
    COMPOSE = "COMPOSE"


@dataclass
class PipelineTask:
    """A single task in the pipeline."""
    task_type: TaskType
    scene_id: str | None = None  # None for project-level tasks like COMPOSE
    depends_on: list[str] = field(default_factory=list)
    task_id: str | None = None  # Celery task ID when submitted
    status: str = "PENDING"
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class PipelineDefinition:
    """Definition of a pipeline to execute.
    
    A pipeline is a DAG of tasks. Each task may depend on others.
    """
    project_id: str
    tasks: list[PipelineTask] = field(default_factory=list)
    
    def add_tts_task(self, scene_id: str) -> PipelineTask:
        """Add a TTS task for a scene."""
        task = PipelineTask(task_type=TaskType.TTS, scene_id=scene_id)
        self.tasks.append(task)
        return task
    
    def add_imagegen_task(self, scene_id: str) -> PipelineTask:
        """Add an image generation task for a scene."""
        task = PipelineTask(task_type=TaskType.IMAGEGEN, scene_id=scene_id)
        self.tasks.append(task)
        return task
    
    def add_videogen_task(self, scene_id: str, depends_on: list[str] | None = None) -> PipelineTask:
        """Add a video generation task for a scene.
        
        Typically depends on TTS and IMAGEGEN tasks for the same scene.
        """
        task = PipelineTask(
            task_type=TaskType.VIDEOGEN,
            scene_id=scene_id,
            depends_on=depends_on or [],
        )
        self.tasks.append(task)
        return task
    
    def add_compose_task(self, depends_on: list[str] | None = None) -> PipelineTask:
        """Add the final composition task.
        
        Typically depends on all VIDEOGEN tasks.
        """
        task = PipelineTask(
            task_type=TaskType.COMPOSE,
            depends_on=depends_on or [],
        )
        self.tasks.append(task)
        return task


def build_full_pipeline(project_id: str, scene_ids: list[str]) -> PipelineDefinition:
    """Build a full pipeline for a project.
    
    Full pipeline includes:
    - TTS for all scenes (parallel)
    - ImageGen for all scenes (parallel)
    - VideoGen for all scenes (each depends on its TTS + ImageGen)
    - Final composition (depends on all VideoGen)
    
    Args:
        project_id: The project ID.
        scene_ids: List of scene IDs in order.
    
    Returns:
        A PipelineDefinition with all tasks set up.
    """
    pipeline = PipelineDefinition(project_id=project_id)
    
    videogen_task_indices: list[int] = []
    
    for scene_id in scene_ids:
        # TTS and ImageGen can run in parallel
        tts_idx = len(pipeline.tasks)
        pipeline.add_tts_task(scene_id)
        
        imagegen_idx = len(pipeline.tasks)
        pipeline.add_imagegen_task(scene_id)
        
        # VideoGen depends on both TTS and ImageGen
        videogen_idx = len(pipeline.tasks)
        pipeline.add_videogen_task(
            scene_id,
            depends_on=[f"task_{tts_idx}", f"task_{imagegen_idx}"],
        )
        videogen_task_indices.append(videogen_idx)
    
    # Final composition depends on all video generation tasks
    pipeline.add_compose_task(
        depends_on=[f"task_{idx}" for idx in videogen_task_indices]
    )
    
    return pipeline


def build_tts_only_pipeline(project_id: str, scene_ids: list[str]) -> PipelineDefinition:
    """Build a TTS-only pipeline."""
    pipeline = PipelineDefinition(project_id=project_id)
    for scene_id in scene_ids:
        pipeline.add_tts_task(scene_id)
    return pipeline


def build_imagegen_only_pipeline(project_id: str, scene_ids: list[str]) -> PipelineDefinition:
    """Build an image generation-only pipeline."""
    pipeline = PipelineDefinition(project_id=project_id)
    for scene_id in scene_ids:
        pipeline.add_imagegen_task(scene_id)
    return pipeline
