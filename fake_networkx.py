import argparse
import numpy as np
import scipy.sparse
from io_util import read_pkl, writepkl

# implement a fake networkx graph like class with npz backend
# assumes fixed number of nodes and edges
# skeletons.nodes()
# skeletons.nodes(data=True)
# skeletons.nodes[n][attr]
# skeletons.edges()
# skeletons.edges[e][attr]
# skeletons.edges(data=True)
# return u, v, data, where data is a dict of edge attributes which can be SET


class FakeNetworkXGraph:
    def __init__(
        self,
        node_attributes,
        edge_attribute,
        node_dtype=np.uint16,
        edge_dtype=np.float32,
    ):
        # node_attributes = ["skeleton_id", "z", "y", "x"]
        # edge_attribute = "length"
        self.node_attributes = sorted(node_attributes)
        self.node_dtype = node_dtype
        # since edges will be saved as 2D dok matrix, can only take single attribute
        assert type(edge_attribute) == str
        self.edge_attribute = edge_attribute
        self.edge_dtype = edge_dtype

        self._nodes = None  # will be saved as [N, #node_attributes] npz
        self._edges = None  # will be saved as dok npz

        self.nodes = None
        self.edges = None

    def init_viewers(self):
        assert self._nodes is not None
        self.nodes = FakeNodeViewer(self._nodes, self.node_attributes)
        assert self._edges is not None
        self.edges = FakeEdgeViewer(self._edges, self.edge_attribute)

    def load_graph(self, graph):
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
        assert len(set([len(nodes[key]) for key in nodes])) == 1

        self._nodes = np.stack(
            [np.array(nodes[key]) for key in self.node_attributes], axis=1
        ).astype(self.node_dtype)

        edges = scipy.sparse.dok_matrix(
            (len(graph.nodes), len(graph.nodes)), dtype=self.edge_dtype
        )
        for edge_0, edge_1, data in graph.edges(data=True):
            edge = tuple(sorted([edge_0, edge_1]))
            edges[edge] = (
                data[self.edge_attribute] if self.edge_attribute in data else -1
            )

        self._edges = edges
        self.init_viewers()

    def load_npz(self, node_npz_file, edge_npz_file):
        self._nodes = np.load(node_npz_file)["data"]
        self._edges = scipy.sparse.load_npz(edge_npz_file).todok()
        self.init_viewers()

    def save_npz(self, node_npz_file, edge_npz_file):
        assert self._nodes is not None
        assert self._edges is not None
        np.savez_compressed(node_npz_file, data=self._nodes)
        scipy.sparse.save_npz(edge_npz_file, self._edges.tocoo())


class FakeNodeViewer:
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


class FakeEdgeViewer:
    def __init__(self, edges, edge_attribute):
        self._edges = edges
        self._edge_attribute = edge_attribute

    def __getitem__(self, key):
        key = tuple(sorted(key))
        return FakeEdgeDataViewer(self._edges, self._edge_attribute, key)

    def __call__(self, data=False):
        indices = self._edges.nonzero()
        if not data:
            return ((i, j) for i, j in zip(indices[0], indices[1]))
        else:
            return ((i, j, self[i, j]) for i, j in zip(indices[0], indices[1]))


class FakeEdgeDataViewer:
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


def patch_pkl(old_pkl, new_pkl):
    gt_graph, node_out = read_pkl(old_pkl)
    fake_graph = FakeNetworkXGraph(["skeleton_id", "z", "y", "x"], "length")
    fake_graph.load_graph(gt_graph)

    writepkl(new_pkl, [fake_graph, node_out])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool to patch gt_stats pkl file")
    parser.add_argument("--old_pkl", type=str, help="old pkl file")
    parser.add_argument("--new_pkl", type=str, help="new pkl file")
    args = parser.parse_args()

    patch_pkl(args.old_pkl, args.new_pkl)
