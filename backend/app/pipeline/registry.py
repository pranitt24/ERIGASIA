from functools import lru_cache
from types import SimpleNamespace

from ..config import settings
from . import mock, real


@lru_cache
def get_pipeline() -> SimpleNamespace:
    """Mix and match: e.g. RETRIEVER_IMPL=real, GENERATOR_IMPL=mock while the model is still training."""
    m, r = mock.build(), real.build()
    pick = lambda name, attr: getattr(r if settings.impl(name) == "real" else m, attr)
    return SimpleNamespace(
        retriever=pick("RETRIEVER", "retriever"),
        generator=pick("GENERATOR", "generator"),
        grader=pick("GRADER", "grader"),
        kb=pick("KB", "kb"),
    )
