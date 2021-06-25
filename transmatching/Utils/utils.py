import igl
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from torch import nn
import copy
from transmatching.Model.debug import Debug
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist


try:
    from pykeops.torch import LazyTensor
except ImportError:
    Debug.keops=False
    
def est_area(X,sigma=1e3):
    if Debug.keops and 4*X.shape[0]*X.shape[1]**2 > 2e9:
        lX = LazyTensor(X[:,None,:,:].contiguous())
        lXt = LazyTensor(X[:,:,None,:].contiguous())
        Ds = ((lX-lXt)**2).sum(-1)
#         Ds = 1/(-sigma*Ds).exp().sum(dim=2).squeeze(-1)
        Ds = 1/(-Ds+0.05**2).step().sum(dim=2).squeeze(-1)
    else:
        Ds = torch.cdist(X,X)
        Ds = 1/(Ds<0.05).float().sum(-1)
#         Ds = 1/(-sigma*Ds).exp().sum(-1) #wrong formula, should be Ds^2
    return Ds    

def chamfer_loss(X,Y):
#     if Debug.keops and 4*X.shape[0]*X.shape[1]*Y.shape[1] > 1e8:
#         print('.')
#         lX = LazyTensor(X[:,None,:,:].contiguous())
#         lXt = LazyTensor(Y[:,:,None,:].contiguous())
#         Ds = ((lX-lXt)**2).sum(-1).sqrt()
#         losses = Ds.min(2).mean(-1) + Ds.min(1).mean(-1)
#     else:
    dist = torch.cdist(X,Y)
    losses = dist.min(-1)[0].mean(-1)+dist.min(-2)[0].mean(-1)
    return losses


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_errors(d, gt_mat):

    p2p = torch.argmin(d, dim=-1).cpu()

    err = np.empty(p2p.shape[0])
    for i in range(p2p.shape[0]):
        pred = p2p[i]
        err[i] = gt_mat[pred, i]

    return err


def get_errors_2s(d1, d2, gt_mat):

    n = d1.shape[0]
    p2p1 = torch.argmin(d1, dim=-1)
    p2p2 = torch.argmin(d2, dim=-1)

    err = np.empty(n)
    for i in range(n):
        pred1 = p2p1[i]
        pred2 = p2p2[pred1]
        err[i] = gt_mat[pred2, i]

    return err



def geo_plot(err):

    err = np.array(err)
    x = np.linspace(0, 1, 1000)
    y = np.mean(err[:, None] < x[None, :], 0)

    plt.plot(x, y)
    plt.grid()
    plt.show()


def plot3d(x):

    fig = go.Figure(data=[go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode='markers')])
    fig.show()


def plot3d_col(x, c):

    fig = go.Figure(data=[go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                                       mode='markers',
                                       marker=dict(color=c, colorscale="viridis", size=5, showscale=True),
    )])
    fig.show()


def save3d_col(x, c, path):

    fig = go.Figure(data=[go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                                       mode='markers',
                                       marker=dict(color=c, colorscale="viridis", size=5, showscale=True),
    )])
    fig.write_image(path, format="png")


def plot_colormap(verts, trivs, cols, colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']]):

    # "Draw multiple triangle meshes side by side"
    if type(verts) is not list:
        verts = [verts]
    if type(trivs) is not list:
        trivs = [trivs]
    if type(cols) is not list:
        cols = [cols]

    nshapes = min([len(verts), len(cols), len(trivs)])

    fig = make_subplots(rows=1, cols=nshapes, specs=[[{'type': 'surface'} for i in range(nshapes)]])
    for i, [vert, triv, col] in enumerate(zip(verts, trivs, cols)):
        if col is not None:
            mesh = go.Mesh3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2],
                             i=triv[:, 0], j=triv[:, 1], k=triv[:, 2],
                             intensity=col,
                             colorscale=colorscale,
                             color='lightpink', opacity=1)
        else:
            mesh = go.Mesh3d(x=vert[:, 0], z=vert[:, 1], y=vert[:, 2],
                             i=triv[:, 0], j=triv[:, 1], k=triv[:, 2])

        fig.add_trace(mesh, row=1, col=i + 1)
        fig.get_subplot(1, i + 1).aspectmode = "data"

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=4, z=-1)
        )
        fig.get_subplot(1, i + 1).camera = camera

    #     fig = go.Figure(data=[mesh], layout=layout)
    fig.update_layout(
        #       autosize=True,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="LightSteelBlue")
    fig.show()
    return fig



def RandomRotateCustom(shape, degree, axis):

    device = shape.device
    degree = np.pi * np.random.uniform(low=-np.abs(degree), high=np.abs(degree)) / 180.0
    sin, cos = np.sin(degree), np.cos(degree)


    if axis == 0:
        matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    elif axis == 1:
        matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    else:
        matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

    return torch.matmul(shape, torch.Tensor(matrix).to(device))

def RandomRotateCustomAllAxis(shape, degree):

    device = shape.device

    degree = np.pi * np.random.uniform(low=-np.abs(degree), high=np.abs(degree)) / 180.0
    sin, cos = np.sin(degree), np.cos(degree)
    matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
    shape = torch.matmul(shape, torch.Tensor(matrix).to(device))

    degree = np.pi * np.random.uniform(low=-np.abs(degree), high=np.abs(degree)) / 180.0
    sin, cos = np.sin(degree), np.cos(degree)
    matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
    shape = torch.matmul(shape, torch.Tensor(matrix).to(device))


    degree = np.pi * np.random.uniform(low=-np.abs(degree), high=np.abs(degree)) / 180.0
    sin, cos = np.sin(degree), np.cos(degree)
    matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

    return torch.matmul(shape, torch.Tensor(matrix).to(device))


def split_shape(shape, threshold=10000):

    n_parts = shape.shape[-2] // threshold
    d = {k: shape[[i for i in range(shape.shape[-2]) if i % n_parts == k]] for k in range(n_parts)}

    return d, n_parts


def put_back_together(d, n_v):

    final = torch.empty(n_v, 3)
    n_parts = len(d.keys())

    for i in range(n_v):
        pick_d = i % n_parts
        pick_idx = i // n_parts
        final[i] = d[pick_d][:, pick_idx, :]

    return final


def approximate_geodesic_distances(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute the geodesic distances approximated by the dijkstra method weighted by
    euclidean edge length
    Args:
        v: the mesh points
        f: the mesh faces
    Returns:
        an nxn matrix which contains the approximated distances
    """

    a = igl.adjacency_matrix(f)
    dist = cdist(v, v)
    values = dist[np.nonzero(a)]
    matrix = sparse.coo_matrix((values, np.nonzero(a)), shape=(v.shape[0], v.shape[0]))
    d = dijkstra(matrix, directed=False)
    return d


def area_weighted_normalization(shape, rescale: bool = True):

    if rescale:
        shape = shape * 0.741

    shape_area = est_area(shape[None, ...])[0]
    shape = shape - (
        shape * (shape_area / shape_area.sum(-1, keepdims=True))[..., None]
    ).sum(-2, keepdims=True)
    return shape


