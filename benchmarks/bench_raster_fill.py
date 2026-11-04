import numpy as np
from landlab.graph.quantity.ext.of_link import calc_midpoint_of_link
from landlab.graph.quantity.ext.of_patch import calc_area_at_patch
from landlab.graph.quantity.ext.of_patch import calc_centroid_at_patch
from landlab.graph.structured_quad.ext.at_cell import fill_node_at_cell
from landlab.graph.structured_quad.ext.at_face import fill_nodes_at_face
from landlab.graph.structured_quad.ext.at_link import fill_nodes_at_link
from landlab.graph.structured_quad.ext.at_link import fill_patches_at_link
from landlab.graph.structured_quad.ext.at_node import fill_link_dirs_at_node
from landlab.graph.structured_quad.ext.at_node import fill_links_at_node
from landlab.graph.structured_quad.ext.at_node import fill_patches_at_node
from landlab.graph.structured_quad.ext.at_patch import fill_links_at_patch
from landlab.graph.structured_quad.structured_quad import UniformRectilinearGraph


def number_of_nodes(shape):
    return shape[0] * shape[1]


def number_of_links(shape):
    return shape[0] * (shape[1] - 1) + (shape[0] - 1) * shape[1]


def number_of_patches(shape):
    return (shape[0] - 1) * (shape[1] - 1)


def number_of_faces(shape):
    return number_of_links((shape[0] - 1, shape[1] - 1))


def number_of_cells(shape):
    return (shape[0] - 2) * (shape[1] - 2)


class TimeGraphQuantity:
    params = [16, 32, 64, 128, 256, 512, 1024, 2048]
    param_names = ["n"]

    def setup(self, n):
        graph = UniformRectilinearGraph((n, n))

        self._nodes_at_link = np.empty((number_of_links((n, n)), 2), dtype=int)
        fill_nodes_at_link((n, n), self._nodes_at_link)
        self._x_of_node = graph.x_of_node.copy()
        self._y_of_node = graph.y_of_node.copy()
        self._nodes_at_patch = graph.nodes_at_patch
        self._links_at_patch = graph.links_at_patch
        self._x_of_link = graph.xy_of_link[:, 0].copy()
        self._y_of_link = graph.xy_of_link[:, 1].copy()

    def time_xy_of_link(self, n):
        out = np.empty((number_of_links((n, n)), 2), dtype=float)
        calc_midpoint_of_link(
            self._nodes_at_link,
            self._x_of_node,
            self._y_of_node,
            out,
        )

    def time_area_of_patch(self, n):
        out = np.empty(number_of_patches((n, n)), dtype=float)
        calc_area_at_patch(
            self._nodes_at_patch,
            self._x_of_node,
            self._y_of_node,
            out,
        )

    def time_centroid_of_patch(self, n):
        out = np.empty((number_of_patches((n, n)), 2), dtype=float)
        calc_centroid_at_patch(
            self._links_at_patch,
            self._x_of_link,
            self._y_of_link,
            out,
        )


class TimeRasterFill:
    params = [16, 32, 64, 128, 256, 512, 1024, 2048]
    param_names = ["n"]

    def time_node_at_cell(self, n):
        out = np.empty(number_of_cells((n, n)), dtype=int)
        fill_node_at_cell((n, n), out)

    def time_nodes_at_face(self, n):
        out = np.empty((number_of_faces((n, n)), 2), dtype=int)
        fill_nodes_at_face((n, n), out)

    def time_patches_at_link(self, n):
        out = np.empty((number_of_links((n, n)), 2), dtype=int)
        fill_patches_at_link((n, n), out)

    def time_nodes_at_link(self, n):
        out = np.empty((number_of_links((n, n)), 2), dtype=int)
        fill_nodes_at_link((n, n), out)

    def time_patches_at_node(self, n):
        out = np.empty((number_of_nodes((n, n)), 4), dtype=int)
        fill_patches_at_node((n, n), out)

    def time_links_at_node(self, n):
        out = np.empty((number_of_nodes((n, n)), 4), dtype=int)
        fill_links_at_node((n, n), out)

    def time_link_dirs_at_node(self, n):
        out = np.empty((number_of_nodes((n, n)), 4), dtype=np.int8)
        fill_link_dirs_at_node((n, n), out)

    def time_links_at_patch(self, n):
        out = np.empty((number_of_patches((n, n)), 4), dtype=int)
        fill_links_at_patch((n, n), out)
