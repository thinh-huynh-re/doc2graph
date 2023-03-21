from typing import List, Optional
from tap import Tap

"""
parser = argparse.ArgumentParser(description="Training")

# init
parser.add_argument(
    "--init", action="store_true", help="download data and prepare folders"
)

# features
parser.add_argument(
    "--add-geom",
    "-addG",
    action="store_true",
    help="add geometrical features to nodes",
)
parser.add_argument(
    "--add-embs",
    "-addT",
    action="store_true",
    help="add textual embeddings to nodes",
)
parser.add_argument(
    "--add-hist",
    "-addH",
    action="store_true",
    help="add histogram of contents to nodes",
)
parser.add_argument(
    "--add-visual",
    "-addV",
    action="store_true",
    help="add visual features to nodes",
)
parser.add_argument(
    "--add-eweights",
    "-addE",
    action="store_true",
    help="add edge features to graphs",
)
# data
parser.add_argument(
    "--src-data",
    type=str,
    default="FUNSD",
    help="which data source to use. It can be FUNSD, PAU or CUSTOM",
)
parser.add_argument(
    "--data-type",
    type=str,
    default="img",
    help="if src-data is CUSTOM, define the data source type: img or pdf.",
)
# graphs
parser.add_argument(
    "--edge-type",
    type=str,
    default="fully",
    help="choose the kind of connectivity in the graph. It can be: fully or knn.",
)
parser.add_argument(
    "--node-granularity",
    type=str,
    default="gt",
    help="choose the granularity of nodes to be used. It can be: gt (if given), ocr (words) or yolo (entities).",
)
parser.add_argument(
    "--num-polar-bins",
    type=int,
    default=8,
    help="number of bins into which discretize the space for edge polar features. It must be a power of 2: Default 8.",
)

# training
parser.add_argument(
    "--model",
    type=str,
    default="e2e",
    help="which model to use, which yaml file to load: e2e, edge or gcn",
)
parser.add_argument(
    "--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU."
)
parser.add_argument("--test", action="store_true", help="skip training")
parser.add_argument(
    "--weights",
    "-w",
    nargs="+",
    type=str,
    default=None,
    help="provide a weights file relative path if testing",
)
"""


class MainArgumentParser(Tap):
    # init
    init: bool = False  # download data and prepare folders

    # features
    add_geom: bool = False  # add geometrical features to nodes
    add_embs: bool = False  # add textual embeddings to nodes
    add_hist: bool = False  # add histograms of contents to nodes
    add_visual: bool = False  # add visual features to nodes
    add_eweights: bool = False  # add edge features to graphs

    # data
    src_data: str = (
        "FUNSD"  # which data source to use. It can be either FUNSD, PAU, or CUSTOM
    )
    data_type: str = (
        "img"  # if src-data is CUSTOM, define the data source type: img or pdf
    )

    # graphs
    edge_type: str = (
        "fully"  # choose the kind of connectivity in the graph. It can be: fully or knn
    )
    node_granularity: str = "gt"  # choose the granularity of nodes to be used. It can be: gt (if given), ocr (words) or yolo (entities)
    num_polar_bins: int = 8  # number of bins into which discretize the space for edge polar features. It must be a power of 2

    # training
    model: str = "e2e"  # which model to use, which yaml file to load: e2e, edge or gcn
    gpu: int = -1  # which GPU to use. Set -1 to use CPU
    test: bool = False  # skip training
    weights: List[str] = []  # provide a weights file relative path if testing
