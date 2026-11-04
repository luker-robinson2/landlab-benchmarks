import numpy as np
from landlab import RasterModelGrid


class TimeRasterGradient:
    params = [16, 32, 64, 128, 256, 512, 1024, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.grid = RasterModelGrid((n, n), (1.0, 2.0))
        self.value_at_node = np.random.uniform(size=self.grid.number_of_nodes)
        self.value_at_link = np.random.uniform(size=self.grid.number_of_links)
        self.out_at_node = self.grid.empty(at="node")
        self.out_at_link = self.grid.empty(at="link")

    def time_grad(self, n):
        self.grid.calc_grad_at_link(self.value_at_node, out=self.out_at_link)

    def time_diff(self, n):
        self.grid.calc_diff_at_link(self.value_at_node, out=self.out_at_link)

    def time_flux_div(self, n):
        self.grid.calc_flux_div_at_node(self.value_at_link, out=self.out_at_node)
