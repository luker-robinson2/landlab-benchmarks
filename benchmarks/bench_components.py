"""One ``run_one_step`` of commonly used components, as grid size grows.

Each benchmark starts from the same seeded terrain: uniform noise on a gentle
slope that drains to the bottom edge, so flow routing has somewhere to go.
"""

import numpy as np
from landlab.components import DepthDependentDiffuser
from landlab.components import ExponentialWeatherer
from landlab.components import FastscapeEroder
from landlab.components import FlowAccumulator
from landlab.components import LinearDiffuser

from landlab import RasterModelGrid

SIZES = [64, 128, 256, 512]


def make_terrain(n):
    rng = np.random.default_rng(1945)
    grid = RasterModelGrid((n, n), xy_spacing=10.0)
    z = grid.add_field(
        "topographic__elevation",
        rng.uniform(size=grid.number_of_nodes) + 0.01 * grid.y_of_node,
        at="node",
    )
    grid.set_closed_boundaries_at_grid_edges(True, True, True, False)
    return grid, z


class TimeFlowAccumulator:
    params = [SIZES, ["D8", "D4"]]
    param_names = ["n", "flow_director"]

    def setup(self, n, flow_director):
        self.grid, _ = make_terrain(n)
        self.fa = FlowAccumulator(self.grid, flow_director=flow_director)

    def time_run_one_step(self, n, flow_director):
        self.fa.run_one_step()


class TimeFastscapeEroder:
    params = [SIZES]
    param_names = ["n"]

    def setup(self, n):
        self.grid, self.z = make_terrain(n)
        FlowAccumulator(self.grid, flow_director="D8").run_one_step()
        self.z0 = self.z.copy()
        self.fsc = FastscapeEroder(self.grid, K_sp=1e-5, m_sp=0.5, n_sp=1.0)

    def time_run_one_step(self, n):
        self.z[:] = self.z0
        self.fsc.run_one_step(1000.0)


class TimeLinearDiffuser:
    params = [SIZES]
    param_names = ["n"]

    def setup(self, n):
        self.grid, _ = make_terrain(n)
        self.ld = LinearDiffuser(self.grid, linear_diffusivity=0.01)

    def time_run_one_step(self, n):
        self.ld.run_one_step(100.0)


class TimeSoilProductionAndTransport:
    params = [SIZES]
    param_names = ["n"]

    def setup(self, n):
        self.grid, z = make_terrain(n)
        self.grid.add_field("bedrock__elevation", z - 0.5, at="node")
        self.grid.add_field(
            "soil__depth", np.full(self.grid.number_of_nodes, 0.5), at="node"
        )
        self.weatherer = ExponentialWeatherer(self.grid)
        self.diffuser = DepthDependentDiffuser(self.grid)

    def time_exponential_weatherer(self, n):
        self.weatherer.run_one_step()

    def time_depth_dependent_diffuser(self, n):
        self.diffuser.run_one_step(10.0)
