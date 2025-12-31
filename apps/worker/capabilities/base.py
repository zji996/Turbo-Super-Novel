from __future__ import annotations

from abc import ABC, abstractmethod


class CapabilityWorker(ABC):
    name: str
    requires_gpu: bool

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def load_models(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def unload_models(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, job: dict) -> dict:
        raise NotImplementedError

