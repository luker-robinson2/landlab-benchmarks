"""Shared helpers for the landlab benchmarks.

Grids
-----
``make_grid(kind, size)`` builds any of the seven landlab grid types at a
named size, so one benchmark can be parametrized over every grid type. Sizes
are named rather than numeric because the grid types can't all hit the same
node count (an icosphere only comes in levels); ``GRID_NODES`` records about
how many nodes each one has.

=========  ==============  ======================================
size       about n_nodes   meant to
=========  ==============  ======================================
small      4,000-10,000    fit in L1/L2 cache
medium     40,000-65,000   fit in the last-level cache
large      160,000-260,000 overflow the last-level cache
=========  ==============  ======================================

Profiles
--------
The ``LANDLAB_BENCH_PROFILE`` environment variable picks which sizes run:
``quick`` (small), ``regular`` (small, medium; the default) or ``full``
(all three).

Skipping
--------
A benchmark that doesn't apply (a grid type an operation doesn't support, a
missing optional dependency) raises ``NotImplementedError`` in ``setup``, so
asv reports it as skipped and every machine runs the same suite.
"""

import os

import numpy as np
from landlab.grid.icosphere import IcosphereGlobalGrid

from landlab import FramedVoronoiGrid
from landlab import HexModelGrid
from landlab import NetworkModelGrid
from landlab import RadialModelGrid
from landlab import RasterModelGrid
from landlab import VoronoiDelaunayGrid

SEED = 1945

GRID_KINDS = [
    "raster",
    "hex",
    "voronoi",
    "framed_voronoi",
    "radial",
    "network",
    "icosphere",
]

_PROFILES = {
    "quick": ["small"],
    "regular": ["small", "medium"],
    "full": ["small", "medium", "large"],
}


def sizes():
    """Return the sizes selected by ``LANDLAB_BENCH_PROFILE``."""
    profile = os.environ.get("LANDLAB_BENCH_PROFILE", "regular")
    try:
        return list(_PROFILES[profile])
    except KeyError:
        raise ValueError(
            f"LANDLAB_BENCH_PROFILE must be one of {', '.join(_PROFILES)} (got {profile!r})"
        ) from None


# side length (raster, hex, framed voronoi), point count (voronoi),
# rings (radial), nodes (network) and level (icosphere) for each size
_SHAPES = {
    "small": {"side": 64, "rings": 37, "level": 5},
    "medium": {"side": 256, "rings": 147, "level": 6},
    "large": {"side": 512, "rings": 295, "level": 7},
}


def make_grid(kind, size, spacing=10.0):
    """Build a grid of type *kind* ("raster", "hex", ...) at a named *size*."""
    side = _SHAPES[size]["side"]
    if kind == "raster":
        return RasterModelGrid((side, side), xy_spacing=spacing)
    if kind == "hex":
        return HexModelGrid((side, side), spacing=spacing, node_layout="rect")
    if kind == "framed_voronoi":
        return FramedVoronoiGrid((side, side), xy_spacing=spacing, seed=SEED)
    if kind == "voronoi":
        rng = np.random.default_rng(SEED)
        n = side * side
        return VoronoiDelaunayGrid(
            rng.uniform(0, side * spacing, n), rng.uniform(0, side * spacing, n)
        )
    if kind == "radial":
        return RadialModelGrid(n_rings=_SHAPES[size]["rings"], spacing=spacing)
    if kind == "network":
        return _make_network(side * side, spacing)
    if kind == "icosphere":
        return IcosphereGlobalGrid(
            radius=6.371e6, mesh_densification_level=_SHAPES[size]["level"]
        )
    raise ValueError(f"unknown grid kind {kind!r}")


def _make_network(n_nodes, spacing):
    """A binary branching network: node i drains to node (i - 1) // 2."""
    child = np.arange(1, n_nodes)
    parent = (child - 1) // 2
    depth = np.floor(np.log2(np.arange(n_nodes) + 1))
    x = np.arange(n_nodes) - 2.0**depth
    return NetworkModelGrid(
        (depth * spacing, x * spacing), links=np.column_stack((child, parent))
    )


def add_terrain(grid, name="topographic__elevation"):
    """Add a seeded rough surface that slopes gently in y."""
    rng = np.random.default_rng(SEED)
    return grid.add_field(
        name,
        rng.uniform(size=grid.number_of_nodes) + 0.01 * grid.y_of_node,
        at="node",
        clobber=True,
    )


def require(condition_or_callable, reason=""):
    """Skip the current benchmark (raise NotImplementedError) unless *condition* holds.

    *condition_or_callable* may be a bool, or a callable that is tried once:
    any exception it raises means "not supported" and skips the benchmark.
    """
    if callable(condition_or_callable):
        try:
            condition_or_callable()
        except Exception as error:
            raise NotImplementedError(reason or repr(error)) from error
    elif not condition_or_callable:
        raise NotImplementedError(reason)
