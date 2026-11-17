"""LeRobot HF feature helpers.

Upstream ``get_hf_features_from_features`` maps every ``shape==(1,)`` field to a
scalar ``Value``. That is correct for ``timestamp`` / index columns, but it
collapses ``observation.*`` width-1 vectors (e.g. env_state zeros) and breaks
ACT batching. Use ``get_hf_features_keep_len1_obs`` when writing or loading
those datasets.
"""

from __future__ import annotations

_UPSTREAM = None


def _upstream():
    """Original LeRobot converter (captured before we patch)."""
    global _UPSTREAM
    if _UPSTREAM is None:
        from lerobot.datasets.utils import get_hf_features_from_features as fn

        _UPSTREAM = fn
    return _UPSTREAM


def get_hf_features_keep_len1_obs(features: dict):
    """Like LeRobot's converter, but keep ``observation.*`` shape (1,) as vectors."""
    import datasets

    keep_vector = any(
        key.startswith("observation.") and tuple(ft.get("shape", ())) == (1,)
        for key, ft in features.items()
    )
    if not keep_vector:
        return _upstream()(features)

    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        if ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif tuple(ft["shape"]) == (1,) and key.startswith("observation."):
            hf_features[key] = datasets.Sequence(
                length=1, feature=datasets.Value(dtype=ft["dtype"])
            )
        elif tuple(ft["shape"]) == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=int(ft["shape"][0]),
                feature=datasets.Value(dtype=ft["dtype"]),
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")
    return datasets.Features(hf_features)


def patch_lerobot_len1_obs_features() -> None:
    """Install the width-1 observation vector patch into LeRobot dataset modules."""
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
    import lerobot.datasets.utils as lerobot_utils

    # Capture original before replacing (idempotent).
    _upstream()
    lerobot_utils.get_hf_features_from_features = get_hf_features_keep_len1_obs
    lerobot_dataset.get_hf_features_from_features = get_hf_features_keep_len1_obs
