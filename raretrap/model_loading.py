"""Explicit loading policy around the unchanged production asset loaders."""
from contextlib import contextmanager


class _LocalCodeLoader:
    def __init__(self, loader):
        self.loader = loader

    def from_pretrained(self, *args, **kwargs):
        if kwargs.get("trust_remote_code") not in (None, False):
            raise ValueError("Remote model Python code is not supported by public commands.")
        return self.loader.from_pretrained(*args, **dict(kwargs, trust_remote_code=False))


@contextmanager
def local_model_code_only(asset_module):
    """Scope the policy to one loader module, restoring it even on failure.

    Public experiments run in separate processes. Do not run concurrent model
    loads in one process: the production factories use module-level bindings.
    No global Transformers class, numerical setting, or RNG state is modified.
    """
    names = ("AutoTokenizer", "AutoModelForCausalLM")
    previous = {name: getattr(asset_module, name) for name in names}
    try:
        for name, loader in previous.items():
            setattr(asset_module, name, _LocalCodeLoader(loader))
        yield
    finally:
        for name, loader in previous.items():
            setattr(asset_module, name, loader)
