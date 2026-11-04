"""Thread scaling of the OpenMP-parallel (``prange``) raster kernels.

The number of OpenMP threads is a benchmark parameter, set with threadpoolctl
so every thread count runs in the same process. On a build without OpenMP
(e.g. the macOS wheels) every thread count runs serially, which is itself
useful to see.
"""

import numpy as np
from threadpoolctl import threadpool_limits

from landlab import RasterModelGrid


class TimeOpenMPRasterKernels:
    params = ([256, 1024, 2048], [1, 2, 4, 8])
    param_names = ["n", "threads"]

    def setup(self, n, threads):
        self._limits = threadpool_limits(limits=threads, user_api="openmp")

        rng = np.random.default_rng(1945)
        self.grid = RasterModelGrid((n, n), xy_spacing=10.0)
        self.value_at_node = rng.uniform(size=self.grid.number_of_nodes)
        self.value_at_link = rng.uniform(size=self.grid.number_of_links)
        self.out_at_node = self.grid.empty(at="node")
        self.out_at_link = self.grid.empty(at="link")

    def teardown(self, n, threads):
        self._limits.restore_original_limits()

    def time_grad_at_link(self, n, threads):
        self.grid.calc_grad_at_link(self.value_at_node, out=self.out_at_link)

    def time_flux_div_at_node(self, n, threads):
        self.grid.calc_flux_div_at_node(self.value_at_link, out=self.out_at_node)

    def time_map_max_of_link_nodes_to_link(self, n, threads):
        self.grid.map_max_of_link_nodes_to_link(
            self.value_at_node, out=self.out_at_link
        )
