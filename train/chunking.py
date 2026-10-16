"""Pure helpers for chunked runs that retain one global training schedule."""


def resolve_stop_at_step(total_steps: int, stop_at_step: int | None,
                         start_step: int = 0) -> int:
    if total_steps <= 0:
        raise ValueError("schedule horizon must be positive")
    stop = total_steps if stop_at_step is None else stop_at_step
    if stop <= start_step:
        raise ValueError("stop_at_step must be after the resumed step")
    if stop > total_steps:
        raise ValueError("stop_at_step cannot exceed the schedule horizon")
    return stop


def chunk_is_final(stop_at_step: int, total_steps: int) -> bool:
    return stop_at_step == total_steps


def validate_schedule_horizon(saved: int | None, requested: int) -> None:
    if saved is not None and saved != requested:
        raise ValueError(
            f"schedule horizon mismatch: checkpoint={saved}, requested={requested}")
