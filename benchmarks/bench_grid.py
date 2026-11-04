"""Grid construction and grid operators, on every grid type.

Building some grid types is slow (a medium framed Voronoi grid takes seconds),
so the operator benchmarks build their grids once in ``setup_cache`` and asv
hands them to ``setup``. An operator a grid type doesn't support is detected
by calling it once in ``setup``; it then shows as skipped.
"""

import numpy as np

from ._common import GRID_KINDS
from ._common import SEED
from ._common import add_terrain
from ._common import make_grid
from ._common import require
from ._common import sizes


class TimeGridCreation:
    params = [GRID_KINDS, sizes()]
    param_names = ["grid", "size"]
    timeout = 300

    def time_create(self, kind, size):
        make_grid(kind, size)

    def peakmem_create(self, kind, size):
        make_grid(kind, size)


class _GridOperator:
    """Time one grid operator, given values at nodes or links, on every grid type."""

    params = [GRID_KINDS, sizes()]
    param_names = ["grid", "size"]
    timeout = 600

    def setup_cache(self):
        rng = np.random.default_rng(SEED)
        grids = {}
        for kind in GRID_KINDS:
            for size in sizes():
                grid = make_grid(kind, size)
                add_terrain(grid)
                grid.add_field(
                    "unit_flux", rng.uniform(size=grid.number_of_links), at="link"
                )
                grids[kind, size] = grid
        return grids

    def setup(self, grids, kind, size):
        self.grid = grids[kind, size]
        self.value_at_node = self.grid.at_node["topographic__elevation"]
        self.value_at_link = self.grid.at_link["unit_flux"]
        self.out_at_node = self.grid.empty(at="node")
        self.out_at_link = self.grid.empty(at="link")
        require(self.run, f"{type(self).__name__} is not supported on a {kind} grid")

    def time_operator(self, grids, kind, size):
        self.run()


class TimeCalcGradAtLink(_GridOperator):
    def run(self):
        self.grid.calc_grad_at_link(self.value_at_node, out=self.out_at_link)


class TimeCalcDiffAtLink(_GridOperator):
    def run(self):
        self.grid.calc_diff_at_link(self.value_at_node, out=self.out_at_link)


class TimeCalcFluxDivAtNode(_GridOperator):
    def run(self):
        self.grid.calc_flux_div_at_node(self.value_at_link, out=self.out_at_node)


class TimeMapMeanOfLinkNodesToLink(_GridOperator):
    def run(self):
        self.grid.map_mean_of_link_nodes_to_link(
            self.value_at_node, out=self.out_at_link
        )


class TimeMapMaxOfLinkNodesToLink(_GridOperator):
    def run(self):
        self.grid.map_max_of_link_nodes_to_link(
            self.value_at_node, out=self.out_at_link
        )


class TimeMapMeanOfLinksToNode(_GridOperator):
    def run(self):
        self.grid.map_mean_of_links_to_node(self.value_at_link, out=self.out_at_node)


class TimeCalcSlopeAtNode(_GridOperator):
    def run(self):
        self.grid.calc_slope_at_node(self.value_at_node)
