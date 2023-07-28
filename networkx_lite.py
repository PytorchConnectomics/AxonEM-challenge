import argparse
import numpy as np
import scipy.sparse as sp
from data_io import read_pkl, write_pkl

# implement a light-weight networkx graph like class with npz backend
# assumes fixed number of nodes and edges
# skeletons.nodes()
# skeletons.nodes(data=True)
# skeletons.nodes[n][attr]
# skeletons.edges()
# skeletons.edges[e][attr]
# skeletons.edges(data=True)
# return u, v, data, where data is a dict of edge attributes which can be SET


# The NetworkXGraphLite class is a lightweight version of the NetworkXGraph class.
class NetworkXGraphLite:
    def __init__(
        self,
        node_attributes=["skeleton_id", "z", "y", "x"],
        edge_attribute="length",
        node_dtype=np.uint16,
        edge_dtype=np.float32,
    ):
        self.node_attributes = sorted(node_attributes)
        self.node_dtype = node_dtype
        # since edges will be saved as 2D dok matrix, can only take single attribute
        assert isinstance(edge_attribute, str)
        self.edge_attribute = edge_attribute
        self.edge_dtype = edge_dtype

        self._nodes = None  # will be saved as [N, #node_attributes] npz
        self._edges = None  # will be saved as dok npz

        self.nodes = None
        self.edges = None

    def init_viewers(self):
        """
        The function initializes viewers for nodes and edges.
        """
        assert self._nodes is not None
        self.nodes = NodeViewerLite(self._nodes, self.node_attributes)
        assert self._edges is not None
        self.edges = EdgeViewerLite(self._edges, self.edge_attribute)

    def load_graph(self, graph):
        """
        The function loads a graph into the object, ensuring that the graph nodes have the same
        attributes and storing the node and edge data in appropriate data structures.

        :param graph: The `graph` parameter is an object that represents a graph. It contains
        information about the nodes and edges of the graph
        """
        assert len(graph.nodes) > 0
        # assert every node has the same attributes
        assert list(graph.nodes) == list(range(len(graph.nodes)))

        nodes = {key: [] for key in self.node_attributes}

        minval = np.inf
        maxval = 0
        for node in graph.nodes:
            node = graph.nodes[node]
            for key in self.node_attributes:
                assert key in node
                nodes[key].append(node[key])
                maxval = max(maxval, node[key])
                minval = min(minval, node[key])
        assert minval >= np.iinfo(self.node_dtype).min
        assert maxval <= np.iinfo(self.node_dtype).max
        assert len({len(nodes[key]) for key in nodes}) == 1

        self._nodes = np.stack(
            [np.array(nodes[key]) for key in self.node_attributes], axis=1
        ).astype(self.node_dtype)

        edges = sp.dok_matrix(
            (len(graph.nodes), len(graph.nodes)), dtype=self.edge_dtype
        )
        for edge_0, edge_1, data in graph.edges(data=True):
            edge = tuple(sorted([edge_0, edge_1]))
            edges[edge] = (
                data[self.edge_attribute]
                if self.edge_attribute in data
                else -1
            )

        self._edges = edges
        self.init_viewers()

    def load_npz(self, node_npz_file, edge_npz_file):
        """
        The function `load_npz` loads node and edge data from npz files and initializes viewers.

        :param node_npz_file: The parameter `node_npz_file` is the file path to the .npz file that
        contains the data for the nodes
        :param edge_npz_file: The `edge_npz_file` parameter is a file path to a NumPy compressed sparse
        matrix file (.npz) that contains the edge data
        """
        self._nodes = np.load(node_npz_file)["data"]
        self._edges = sp.load_npz(edge_npz_file).todok()
        self.init_viewers()

    def save_npz(self, node_npz_file, edge_npz_file):
        assert self._nodes is not None
        assert self._edges is not None
        np.savez_compressed(node_npz_file, data=self._nodes)
        sp.save_npz(edge_npz_file, self._edges.tocoo())


# The NodeViewerLite class is a simplified version of a node viewer.
class NodeViewerLite:
    def __init__(self, nodes, node_attributes):
        self._nodes = nodes
        self._node_attributes = node_attributes

    def __getitem__(self, key):
        node = self._nodes[key]
        return {key: node[i] for i, key in enumerate(self._node_attributes)}

    def __call__(self, data=False):
        if not data:
            return range(len(self._nodes))
        else:
            # return generator, not instantiated list
            return ((i, self[i]) for i in range(len(self._nodes)))


# The EdgeViewerLite class is a lightweight viewer for displaying edges.
class EdgeViewerLite:
    def __init__(self, edges, edge_attribute):
        self._edges = edges
        self._edge_attribute = edge_attribute

    def __getitem__(self, key):
        key = tuple(sorted(key))
        return EdgeDataViewerLite(self._edges, self._edge_attribute, key)

    def __call__(self, data=False):
        indices = self._edges.nonzero()
        if not data:
            return ((i, j) for i, j in zip(indices[0], indices[1]))
        else:
            return ((i, j, self[i, j]) for i, j in zip(indices[0], indices[1]))


# The EdgeDataViewerLite class is a lightweight viewer for edge data.
class EdgeDataViewerLite:
    def __init__(self, edges, edge_attribute, key):
        self._edges = edges
        self._edge_attribute = edge_attribute
        self._key = key

    def __getitem__(self, edge_attribute):
        assert edge_attribute == self._edge_attribute
        return self._edges[self._key]

    def __setitem__(self, edge_attribute, value):
        assert edge_attribute == self._edge_attribute
        self._edges[self._key] = value


def convert_networkx_to_lite(networkx_graph):
    """
    The function converts a NetworkX graph to a NetworkXGraphLite graph.

    :param networkx_graph: The `networkx_graph` parameter is a graph object from the NetworkX library.
    It represents a graph with nodes and edges, where each node can have attributes and each edge can
    have attributes
    :return: a NetworkXGraphLite object.
    """
    networkx_lite_graph = NetworkXGraphLite(
        ["skeleton_id", "z", "y", "x"], "length"
    )
    networkx_lite_graph.load_graph(networkx_graph)
    return networkx_lite_graph


def patch_axonEM_stats(old_pkl, new_pkl):
    """
    The function `patch_axonEM_stats` reads a pickle file, converts a graph to a lite version, and
    writes the updated data to a new pickle file.

    :param old_pkl: The path to the old pickle file that contains the data to be patched
    :param new_pkl: The `new_pkl` parameter is the name or path of the new pickle file that you want to
    create or overwrite
    """
    gt_graph, node_out = read_pkl(old_pkl)
    gt_graph_lite = convert_networkx_to_lite(gt_graph)
    write_pkl(new_pkl, [gt_graph_lite, node_out])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool to patch gt_stats pkl file"
    )
    parser.add_argument("--old_pkl", type=str, help="old pkl file")
    parser.add_argument("--new_pkl", type=str, help="new pkl file")
    args = parser.parse_args()

    patch_axonEM_stats(args.old_pkl, args.new_pkl)
