from typing import TypeVar

from openhands.app_server.event_callback.event_callback_models import (
    EventCallbackProcessor,
)

T = TypeVar("T", bound=EventCallbackProcessor)

REGISTERED_EVENT_CALLBACK_PROCESSORS: dict[str, type[EventCallbackProcessor]] = {}


def register_event_callback_processor(cls: T) -> T:
    REGISTERED_EVENT_CALLBACK_PROCESSORS[cls.__name__] = cls
    # Force rebuild of discriminated union so Pydantic sees the new type
    EventCallbackProcessor.model_rebuild(force=True)
    return cls
