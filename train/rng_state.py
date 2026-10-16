"""Checkpoint and isolate the RNG streams used by training."""


def capture_rng_state(torch):
    state = {"cpu": torch.get_rng_state().cpu()}
    if torch.cuda.is_available() and torch.cuda.device_count():
        state["cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    else:
        state["cuda"] = None
    return state


def restore_rng_state(torch, state):
    if not state or state.get("cpu") is None:
        return False
    torch.set_rng_state(state["cpu"].cpu())
    cuda_states = state.get("cuda")
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count differs from runtime")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    return True


def make_generator(torch, device, seed: int):
    return torch.Generator(device=device).manual_seed(seed)
