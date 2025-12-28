"""Novel Project API routes."""

from __future__ import annotations

import uuid
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from libs.dbcore import (
    NovelProject,
    NovelScene,
    NovelPipeline,
    ensure_schema,
    session_scope,
)
from libs.novelcore import parse_text_to_scenes, SceneData

router = APIRouter(prefix="/v1/novel", tags=["novel"])


# ============================================================================
# Request/Response Models
# ============================================================================


class CreateProjectRequest(BaseModel):
    """Request to create a new project."""
    name: str = Field(..., min_length=1, max_length=255)
    source_text: str = Field(..., min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
    scene_delimiter: str = Field(default="---", description="Delimiter to split scenes")
    auto_generate_image_prompts: bool = Field(default=True)


class UpdateProjectRequest(BaseModel):
    """Request to update project configuration."""
    name: str | None = None
    config: dict[str, Any] | None = None


class SceneResponse(BaseModel):
    """Response for a single scene."""
    id: str
    sequence: int
    text: str
    image_prompt: str | None
    status: str
    audio_url: str | None = None
    image_url: str | None = None
    video_url: str | None = None


class ProjectResponse(BaseModel):
    """Response for a project."""
    id: str
    name: str
    status: str
    source_text: str
    config: dict[str, Any]
    scene_count: int
    output_url: str | None = None
    created_at: str
    updated_at: str


class ProjectListResponse(BaseModel):
    """Response for project listing."""
    projects: list[ProjectResponse]
    total: int


class StartPipelineRequest(BaseModel):
    """Request to start a pipeline."""
    pipeline_type: str = Field(default="FULL", pattern="^(FULL|TTS_ONLY|IMAGEGEN_ONLY|VIDEOGEN_ONLY)$")


class PipelineResponse(BaseModel):
    """Response for a pipeline."""
    id: str
    project_id: str
    pipeline_type: str
    status: str
    total_tasks: int
    completed_tasks: int
    started_at: str
    completed_at: str | None
    error: str | None


# ============================================================================
# Project Endpoints
# ============================================================================


@router.post("/projects", response_model=ProjectResponse)
async def create_project(request: CreateProjectRequest) -> dict:
    """Create a new novel video project.
    
    Parses the source text into scenes based on the delimiter.
    """
    ensure_schema()
    
    # Parse text into scenes
    scenes = parse_text_to_scenes(
        request.source_text,
        delimiter=request.scene_delimiter,
        auto_generate_image_prompt=request.auto_generate_image_prompts,
    )
    
    if not scenes:
        raise HTTPException(
            status_code=400,
            detail="No scenes found after parsing. Check your text and delimiter.",
        )
    
    project_id = uuid4()
    
    with session_scope() as session:
        # Create project
        project = NovelProject(
            id=project_id,
            name=request.name,
            status="READY",
            source_text=request.source_text,
            config=request.config,
        )
        session.add(project)
        
        # Create scenes
        for scene_data in scenes:
            scene = NovelScene(
                id=uuid4(),
                project_id=project_id,
                sequence=scene_data.sequence,
                text=scene_data.text,
                image_prompt=scene_data.image_prompt,
                status="PENDING",
            )
            session.add(scene)
        
        session.flush()
        
        return {
            "id": str(project.id),
            "name": project.name,
            "status": project.status,
            "source_text": project.source_text,
            "config": project.config,
            "scene_count": len(scenes),
            "output_url": None,
            "created_at": project.created_at.isoformat() if project.created_at else "",
            "updated_at": project.updated_at.isoformat() if project.updated_at else "",
        }


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(limit: int = 20, offset: int = 0) -> dict:
    """List all projects."""
    ensure_schema()
    
    with session_scope() as session:
        from sqlalchemy import select, func
        
        # Get total count
        total = session.scalar(select(func.count(NovelProject.id)))
        
        # Get projects
        stmt = (
            select(NovelProject)
            .order_by(NovelProject.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        projects = session.scalars(stmt).all()
        
        # Get scene counts
        project_responses = []
        for p in projects:
            scene_count = session.scalar(
                select(func.count(NovelScene.id)).where(NovelScene.project_id == p.id)
            )
            project_responses.append({
                "id": str(p.id),
                "name": p.name,
                "status": p.status,
                "source_text": p.source_text[:200] + "..." if len(p.source_text) > 200 else p.source_text,
                "config": p.config,
                "scene_count": scene_count or 0,
                "output_url": None,
                "created_at": p.created_at.isoformat() if p.created_at else "",
                "updated_at": p.updated_at.isoformat() if p.updated_at else "",
            })
        
        return {
            "projects": project_responses,
            "total": total or 0,
        }


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str) -> dict:
    """Get a project by ID."""
    ensure_schema()
    
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    
    with session_scope() as session:
        from sqlalchemy import select, func
        
        project = session.get(NovelProject, pid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        
        scene_count = session.scalar(
            select(func.count(NovelScene.id)).where(NovelScene.project_id == pid)
        )
        
        return {
            "id": str(project.id),
            "name": project.name,
            "status": project.status,
            "source_text": project.source_text,
            "config": project.config,
            "scene_count": scene_count or 0,
            "output_url": None,
            "created_at": project.created_at.isoformat() if project.created_at else "",
            "updated_at": project.updated_at.isoformat() if project.updated_at else "",
        }


@router.delete("/projects/{project_id}")
async def delete_project(project_id: str) -> dict:
    """Delete a project and its scenes."""
    ensure_schema()
    
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    
    with session_scope() as session:
        from sqlalchemy import delete
        
        project = session.get(NovelProject, pid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Delete scenes first
        session.execute(delete(NovelScene).where(NovelScene.project_id == pid))
        # Delete pipelines
        session.execute(delete(NovelPipeline).where(NovelPipeline.project_id == pid))
        # Delete project
        session.delete(project)
        
        return {"status": "deleted", "id": project_id}


@router.get("/projects/{project_id}/scenes", response_model=list[SceneResponse])
async def get_project_scenes(project_id: str) -> list[dict]:
    """Get all scenes for a project."""
    ensure_schema()
    
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    
    with session_scope() as session:
        from sqlalchemy import select
        
        project = session.get(NovelProject, pid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        
        stmt = (
            select(NovelScene)
            .where(NovelScene.project_id == pid)
            .order_by(NovelScene.sequence)
        )
        scenes = session.scalars(stmt).all()
        
        return [
            {
                "id": str(s.id),
                "sequence": s.sequence,
                "text": s.text,
                "image_prompt": s.image_prompt,
                "status": s.status,
                "audio_url": None,  # TODO: Generate presigned URLs
                "image_url": None,
                "video_url": None,
            }
            for s in scenes
        ]


# ============================================================================
# Pipeline Endpoints
# ============================================================================


@router.post("/projects/{project_id}/pipelines", response_model=PipelineResponse)
async def start_pipeline(project_id: str, request: StartPipelineRequest) -> dict:
    """Start a pipeline for a project.
    
    This submits all necessary tasks to the Celery worker.
    """
    ensure_schema()
    
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    
    with session_scope() as session:
        from sqlalchemy import select, func
        
        project = session.get(NovelProject, pid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if project.status == "PROCESSING":
            raise HTTPException(status_code=400, detail="Project is already being processed")
        
        # Count scenes
        scene_count = session.scalar(
            select(func.count(NovelScene.id)).where(NovelScene.project_id == pid)
        )
        
        if not scene_count:
            raise HTTPException(status_code=400, detail="Project has no scenes")
        
        # Calculate total tasks based on pipeline type
        if request.pipeline_type == "FULL":
            total_tasks = scene_count * 3 + 1  # TTS + ImageGen + VideoGen per scene + Compose
        elif request.pipeline_type in ("TTS_ONLY", "IMAGEGEN_ONLY", "VIDEOGEN_ONLY"):
            total_tasks = scene_count
        else:
            total_tasks = scene_count
        
        # Create pipeline record
        pipeline_id = uuid4()
        pipeline = NovelPipeline(
            id=pipeline_id,
            project_id=pid,
            pipeline_type=request.pipeline_type,
            status="PENDING",
            total_tasks=total_tasks,
            completed_tasks=0,
        )
        session.add(pipeline)
        
        # Update project status
        project.status = "PROCESSING"
        
        session.flush()
        
        # TODO: Submit to Celery
        # celery_app.send_task(
        #     "novel.pipeline.run",
        #     kwargs={
        #         "pipeline_id": str(pipeline_id),
        #         "project_id": project_id,
        #         "pipeline_type": request.pipeline_type,
        #     }
        # )
        
        return {
            "id": str(pipeline.id),
            "project_id": str(pipeline.project_id),
            "pipeline_type": pipeline.pipeline_type,
            "status": pipeline.status,
            "total_tasks": pipeline.total_tasks,
            "completed_tasks": pipeline.completed_tasks,
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else "",
            "completed_at": None,
            "error": None,
        }


@router.get("/projects/{project_id}/pipelines", response_model=list[PipelineResponse])
async def list_project_pipelines(project_id: str) -> list[dict]:
    """List all pipelines for a project."""
    ensure_schema()
    
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    
    with session_scope() as session:
        from sqlalchemy import select
        
        stmt = (
            select(NovelPipeline)
            .where(NovelPipeline.project_id == pid)
            .order_by(NovelPipeline.started_at.desc())
        )
        pipelines = session.scalars(stmt).all()
        
        return [
            {
                "id": str(p.id),
                "project_id": str(p.project_id),
                "pipeline_type": p.pipeline_type,
                "status": p.status,
                "total_tasks": p.total_tasks,
                "completed_tasks": p.completed_tasks,
                "started_at": p.started_at.isoformat() if p.started_at else "",
                "completed_at": p.completed_at.isoformat() if p.completed_at else None,
                "error": p.error,
            }
            for p in pipelines
        ]


@router.get("/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str) -> dict:
    """Get a pipeline by ID."""
    ensure_schema()
    
    try:
        pid = uuid.UUID(pipeline_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid pipeline ID")
    
    with session_scope() as session:
        pipeline = session.get(NovelPipeline, pid)
        if pipeline is None:
            raise HTTPException(status_code=404, detail="Pipeline not found")
        
        return {
            "id": str(pipeline.id),
            "project_id": str(pipeline.project_id),
            "pipeline_type": pipeline.pipeline_type,
            "status": pipeline.status,
            "total_tasks": pipeline.total_tasks,
            "completed_tasks": pipeline.completed_tasks,
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else "",
            "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
            "error": pipeline.error,
        }
