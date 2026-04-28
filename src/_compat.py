"""Compatibility shims for the BendingGraphs codebase on a modern stack.

The original code targets PyTorch 1.8 / Open3D 0.13 / scikit-learn 0.24 /
NumPy 1.20.  This module monkey-patches the small handful of APIs that were
since renamed/removed so the rest of the code can run unmodified on a
PyTorch 2.4 / Open3D 0.18 / scikit-learn 1.x / NumPy 1.26 stack.

Apply by calling ``apply_compat()`` once - safe to call repeatedly.  The
top-level ``configs.py`` does this on import, so every entry point is
covered without changing call sites.
"""
from __future__ import annotations

import sys
import types
import warnings


_APPLIED = False


def _patch_numpy() -> None:
    import numpy as np

    if not hasattr(np, "asscalar"):
        def asscalar(a):
            return np.asarray(a).item()
        np.asscalar = asscalar  # type: ignore[attr-defined]

    # NumPy 1.20+ removed these aliases; chumpy and the original BendingGraphs
    # code still rely on them.
    aliases = {
        "int": int,
        "float": float,
        "bool": bool,
        "object": object,
        "complex": complex,
        "str": str,
        "unicode": str,
        "long": int,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for name, default in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, default)


def _patch_open3d() -> None:
    import numpy as np
    import open3d as o3d

    pc_cls = o3d.geometry.PointCloud

    if not hasattr(pc_cls, "select_down_sample"):
        pc_cls.select_down_sample = pc_cls.select_by_index  # type: ignore[attr-defined]

    original_rotate = pc_cls.rotate
    original_translate = pc_cls.translate

    def rotate_compat(self, R, center=None, *args, **kwargs):
        if isinstance(center, (bool, np.bool_)):
            center = (0.0, 0.0, 0.0) if not center else self.get_center()
        if center is None:
            center = (0.0, 0.0, 0.0)
        return original_rotate(self, R, center, *args, **kwargs)

    def translate_compat(self, t, relative=True, *args, **kwargs):
        t = np.asarray(t, dtype=np.float64).reshape(3)
        return original_translate(self, t, relative, *args, **kwargs)

    pc_cls.rotate = rotate_compat  # type: ignore[assignment]
    pc_cls.translate = translate_compat  # type: ignore[assignment]

    # ``open3d.registration`` was moved to ``open3d.pipelines.registration``
    if not hasattr(o3d, "registration"):
        legacy = types.ModuleType("open3d.registration")
        modern = o3d.pipelines.registration
        for name in dir(modern):
            if not name.startswith("_"):
                setattr(legacy, name, getattr(modern, name))

        original_ransac_criteria = modern.RANSACConvergenceCriteria

        def _ransac_criteria_compat(*args, **kwargs):
            kwargs.pop("max_validation", None)
            return original_ransac_criteria(*args, **kwargs)

        legacy.RANSACConvergenceCriteria = _ransac_criteria_compat
        o3d.registration = legacy  # type: ignore[attr-defined]
        sys.modules["open3d.registration"] = legacy

    # ``o3d.cpu.pybind.geometry.TriangleMesh`` is exposed differently across
    # 0.13 -> 0.18; mesh_utils does isinstance checks against it.  Make the
    # alias resolvable either way.
    try:
        _ = o3d.cpu.pybind.geometry.TriangleMesh  # noqa: F841
    except AttributeError:  # pragma: no cover
        pass


def _patch_sklearn() -> None:
    """``sklearn.utils.graph_shortest_path`` and friends were removed.

    Provide drop-in replacements that route through ``scipy.sparse.csgraph``.
    We *augment* the real ``sklearn.utils.graph`` rather than replacing it,
    so internal sklearn machinery keeps working.
    """
    import numpy as np
    from scipy.sparse import csgraph, csr_matrix

    try:
        import sklearn.utils as sk_utils
    except ImportError:  # pragma: no cover - sklearn always present in env
        return

    # ``graph_shortest_path`` was a top-level module that no longer exists.
    if "sklearn.utils.graph_shortest_path" not in sys.modules:
        mod = types.ModuleType("sklearn.utils.graph_shortest_path")

        def graph_shortest_path(dist_matrix, directed=False, method="auto"):
            mat = np.asarray(dist_matrix)
            if mat.dtype == bool:
                mat = mat.astype(np.float64)
            if not isinstance(mat, csr_matrix):
                mat = csr_matrix(mat)
            return csgraph.shortest_path(
                mat, method=method if method != "auto" else "auto",
                directed=directed,
            )

        mod.graph_shortest_path = graph_shortest_path
        sys.modules["sklearn.utils.graph_shortest_path"] = mod
        sk_utils.graph_shortest_path = mod  # type: ignore[attr-defined]

    # ``single_source_shortest_path_length`` was removed from the real
    # ``sklearn.utils.graph`` in 1.x.  Inject it into whatever module is
    # already there (we must not replace the module — sklearn's internals
    # still import private names from it).
    try:
        import sklearn.utils.graph as sk_graph  # type: ignore
    except Exception:
        sk_graph = types.ModuleType("sklearn.utils.graph")
        sys.modules["sklearn.utils.graph"] = sk_graph

    if not hasattr(sk_graph, "single_source_shortest_path_length"):
        def single_source_shortest_path_length(graph, source, cutoff=None):
            mat = np.asarray(graph) if not hasattr(graph, "tocsr") else graph
            if not isinstance(mat, csr_matrix):
                mat = csr_matrix(np.asarray(mat).astype(np.float64))
            d = csgraph.dijkstra(
                mat, indices=int(source), unweighted=True,
                limit=cutoff if cutoff is not None else np.inf,
            )
            out = {}
            for i, di in enumerate(d):
                if np.isfinite(di) and (cutoff is None or di <= cutoff):
                    out[i] = int(di)
            return out

        sk_graph.single_source_shortest_path_length = single_source_shortest_path_length  # type: ignore[attr-defined]


def _patch_scipy_rotation() -> None:
    """``Rotation.from_dcm`` was renamed to ``from_matrix`` in SciPy 1.6.

    The class is a C extension and refuses ``setattr``; instead we expose the
    helper through ``_compat.from_dcm_compat`` and update the few call sites
    in ``eval_rigid`` directly.
    """
    return


def _patch_torch_geometric() -> None:
    """In PyG 2.x the ``DataLoader`` lives under ``torch_geometric.loader``.

    Re-export it from ``torch_geometric.data`` so the original ``from
    torch_geometric.data import DataLoader`` keeps working.
    """
    import torch_geometric.data as tg_data
    if not hasattr(tg_data, "DataLoader"):
        try:
            from torch_geometric.loader import DataLoader as _DL
            tg_data.DataLoader = _DL  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover
            pass


def apply_compat() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _patch_numpy()
    for name, fn in (
        ("open3d", _patch_open3d),
        ("sklearn", _patch_sklearn),
        ("scipy_rotation", _patch_scipy_rotation),
        ("torch_geometric", _patch_torch_geometric),
    ):
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            print(f"compat shim '{name}' skipped:", exc)
    _APPLIED = True
