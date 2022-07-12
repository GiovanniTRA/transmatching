import os
from pathlib import Path
from typing import Optional, Union

import dotenv
import git
import hydra
import numpy as np
import omegaconf
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose
from matplotlib import pyplot as plt
from plotly.graph_objs import Layout

from transmatching.Model.debug import Debug

try:
    from pykeops.torch import LazyTensor
except ImportError:
    Debug.keops = False


def cdist_argmin(input1, input2, argmin_axis: int):
    """
    Computes cdist + argmin along some axis in chunks of maximum 100 elements

    Args:
        input1: tensor of shape [n, d] to be chunked
        input2: tensor of shape [m, d]
        argmin_axis: the axis of the argmin
    Returns
        A tensor equivalent to torch.cdist(input1, input2).argmin(argmin_axis)
    """
    assert len(input1.shape) == 2
    assert len(input2.shape) == 2
    assert argmin_axis == 1 or argmin_axis == 0
    if argmin_axis == 0:
        input1, input2 = input2, input1

    num_chunks = input1.shape[0] // 100 if 100 < input1.shape[0] else 1
    chunks = input1.chunk(num_chunks)

    argmins = []
    for input1_chunk in chunks:
        argmins.append(torch.cdist(input1_chunk, input2).argmin(-1))
    dists = torch.cat(argmins)
    return dists


def keops_cdist_argmin(X, Y, argmin_axis: int = 0):
    """

    Args:
        X: [b, n, d]
        Y: [b, m, d]
        argmin_axis: int

    Returns:

    """
    assert len(X.shape) == 2
    assert len(Y.shape) == 2

    if Debug.keops:
        if argmin_axis == 1:
            X, Y = Y, X
        lX = LazyTensor(X[None, None, :, :].contiguous())
        lY = LazyTensor(Y[None, :, None, :].contiguous())
        Ds: LazyTensor = ((lX - lY) ** 2).sum(-1)
        return Ds.argKmin(K=1, axis=2).squeeze(-1)
    else:
        return cdist_argmin(X, Y, argmin_axis=argmin_axis)


def cdist_min(input1, input2, min_axis: int):
    """
    Computes cdist + min along some axis in chunks of maximum 100 elements

    Args:
        input1: tensor of shape [n, d] to be chunked
        input2: tensor of shape [m, d]
        min_axis: the axis of the min
    Returns
        A tensor equivalent to torch.cdist(input1, input2).min(argmin_axis)
    """
    assert min_axis == 1 or min_axis == 0
    if min_axis == 0:
        input1, input2 = input2, input1

    num_chunks = input1.shape[0] // 100 if 100 < input1.shape[0] else 1
    chunks = input1.chunk(num_chunks)

    mins = []
    for input1_chunk in chunks:
        mins.append(torch.cdist(input1_chunk, input2).min(-1)[0])
    dists = torch.cat(mins)
    return dists


def chamfer_chunked(y_hat, src):
    assert len(y_hat.shape) == 2, y_hat.shape
    assert len(src.shape) == 2, src.shape
    dists_0 = cdist_min(y_hat, src, 0)
    dists_1 = cdist_min(y_hat, src, 1)

    loss = dists_0.mean(-1) + dists_1.mean(-1)
    return loss


def plot3d(
    x: Union[np.ndarray, torch.Tensor], c: Union[np.ndarray, torch.Tensor]
) -> None:
    """
    Plot the function c over the point cloud x
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x[:, 0],
                y=x[:, 1],
                z=x[:, 2],
                mode="markers",
                marker=dict(color=c, colorscale="viridis", size=5, showscale=True),
            )
        ],
        layout=Layout(scene=dict(aspectmode="data")),
    )
    fig.show()


def calc_tri_areas(vert, triv):
    v1 = vert[triv[:, 0], :]
    v2 = vert[triv[:, 1], :]
    v3 = vert[triv[:, 2], :]

    v1 = v1 - v3
    v2 = v2 - v3

    areas = np.linalg.norm(np.cross(v1, v2), ord=2, axis=-1) * 0.5
    return areas


def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.

    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable

    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.

    It is possible to define all the system specific variables in the `env_file`.

    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environment variables
load_envs()

# Set the cwd to the project root
try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

os.chdir(PROJECT_ROOT)


def invert_permutation(p: np.ndarray, num_elements) -> np.ndarray:
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty_like(p)
    s[p] = np.arange(num_elements)
    return s


def get_dists() -> np.ndarray:
    """
    Get the geodesics distances matrix

    Returns:
        the global geodesic distances matrix
    """
    geo_dists = np.load(
        str(
            PROJECT_ROOT / "evaluation" / "data_aux" / "gt_distances_plain" / "data.npy"
        )
    )
    geo_dists /= np.max(geo_dists)
    return geo_dists


def get_hydra_cfg(config_name: str = "default") -> omegaconf.DictConfig:
    """
    Instantiate and return the hydra config -- streamlit and jupyter compatible

    Args:
        config_name: .yaml configuration name, without the extension

    Returns:
        The desired omegaconf.DictConfig
    """
    GlobalHydra.instance().clear()
    hydra.experimental.initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "evaluation" / "conf")
    )
    return compose(config_name=config_name)


def get_point_colors(
    points: np.ndarray, frequency: float = np.pi, rgb_rescale: bool = True
) -> np.ndarray:
    """
    Create RGB colors for each point in points, using their coordingates
    Args:
        points: the points to color
        frequency: the frequency of oscillation of the oclors
        rgb_rescale: whether the RGB should be rescaled from [0, 1] to [0, 255]

    Returns:
        an ndarray [n, 3] containing one [0, 1] rgb color for each point. If
        rgb_rescale is True the color is in [0, 255]
    """
    colors = (points - points.min(axis=0)[0]) / (
        points.max(axis=0)[0] - points.min(axis=0)[0]
    )
    colors = np.cos(frequency * colors)
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    if rgb_rescale:
        colors *= 255
    return colors


##### UGLY PLOTTER TO ORGANIZE


from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import meshio
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib import cm
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


class Data:
    pass


class Mesh:
    def __init__(
        self,
        *,
        v: np.ndarray,
        f: np.ndarray,
        color: Optional[np.ndarray] = None,
        name: str = None,
    ):
        """
        Utility class to represent a mesh
        :param v: the vertices
        :param f: the faces
        :param color:
        :param name:
        """
        self.v = v
        self.f = f
        if f is not None:
            if f.min() == 1:
                self.f = self.f - 1
        self.color = color
        self.name = name


def _as_mesh(data: Union[Mesh, Data]) -> Mesh:
    """
    Uniform the mesh representation "Data" from pytorch_geometric to the Mesh class
    """
    if isinstance(data, Mesh):
        return data
    elif isinstance(data, meshio.Mesh):
        return Mesh(v=data.points, f=data.cells_dict["triangle"])
    elif isinstance(data, Data):
        return Mesh(
            v=np.asarray(data.pos),
            f=np.asarray(data.face.T),
        )
    elif isinstance(data, np.ndarray):
        return Mesh(v=data, f=None)
    else:
        raise ValueError(f"Data type not understood: <{data.__class__}>")


def _default_layout(fig: go.Figure) -> Figure:
    """
    Set the default camera parameters for the plotly Mesh3D
    :param fig: the figure to adjust
    :return: the adjusted figure
    """
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-0.25, y=0.25, z=2),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        # title='ex',
        scene_aspectmode="auto",
    )

    return fig


def _tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))


def _map_z2color(zval, colormap, vmin, vmax):
    # map the normalized value zval to a corresponding color in the colormap

    if vmin > vmax:
        raise ValueError("incorrect relation between vmin and vmax")
    t = (zval - vmin) / float((vmax - vmin))  # normalize val
    R, G, B, alpha = colormap(t)
    return (
        "rgb("
        + "{:d}".format(int(R * 255 + 0.5))
        + ","
        + "{:d}".format(int(G * 255 + 0.5))
        + ","
        + "{:d}".format(int(B * 255 + 0.5))
        + ")"
    )


def _plotly_trisurf(vertices, faces, colormap=cm.RdBu):
    # x, y, z are lists of coordinates of the triangle vertices
    # faces are the faces that define the triangularization;
    # faces  is a numpy array of shape (no_triangles, 3)
    # insert here the  type check for input data

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    points3D = np.vstack((x, y, z)).T
    tri_vertices = map(
        lambda index: points3D[index], faces
    )  # vertices of the surface triangles
    zmean = [
        np.mean(tri[:, 2]) for tri in tri_vertices
    ]  # mean values of z-coordinates of
    # triangle vertices
    min_zmean = np.min(zmean)
    max_zmean = np.max(zmean)
    facecolor = [_map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]
    I, J, K = _tri_indices(faces)

    triangles = go.Mesh3d(x=x, y=y, z=z, facecolor=facecolor, i=I, j=J, k=K, name="")

    return triangles
    # Plot EDGE - not working
    #     # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
    #     # None separates data corresponding to two consecutive triangles
    #     # lists_coord = [
    #     #     [[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices]
    #     #     for c in range(3)
    #     # ]
    #     # tri_points = tri_vertices
    #     Xe = []
    #     Ye = []
    #     Ze = []
    #     for T in tri_vertices:
    #         Xe.extend([T[k % 3][0] for k in range(4)] + [None])
    #         Ye.extend([T[k % 3][1] for k in range(4)] + [None])
    #         Ze.extend([T[k % 3][2] for k in range(4)] + [None])
    #
    #     # define the trace for triangle sides
    #     lines = go.Scatter3d(
    #         x=Xe,
    #         y=Ye,
    #         z=Ze,
    #         mode="lines",
    #         name="",
    #         line=dict(color="rgb(70,70,70)", width=1),
    #     )
    #     return [triangles, lines]


def _plot_mesh(
    m: Union[Mesh, Data],
    showtriangles: bool = True,
    showscale=False,
    colorscale="Viridis",
    reversescale=False,
    cmax=None,
    cmin=None,
    **kwargs,
) -> Union[go.Mesh3d, go.Scatter3d]:
    """
    Plot the mesh in a plotly graph object
    :param m: the mesh to plot
    :param kwargs: possibly additional parameters for the go.Mesh3D class
    :return: the plotted mesh
    """
    if colorscale is None:
        colorscale = "Viridis"

    m = _as_mesh(m)
    vertices = m.v.astype(np.float64)
    if m.f is not None:
        if showtriangles:
            return _plotly_trisurf(vertices, m.f)
        else:
            faces = m.f.astype(np.uint32)
            return go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                colorscale=colorscale,
                opacity=1,
                intensity=m.color if m.color is not None else vertices[:, 0],
                showscale=showscale,
                reversescale=reversescale,
                cmax=cmax,
                cmin=cmin,
                **kwargs,
            )
    else:
        return go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=m.color
                if m.color is not None
                else vertices[:, 0],  # set color to an array/list of desired values
                colorscale=colorscale,  # choose a colorscale
                opacity=1,
                reversescale=reversescale,
                cmax=cmax,
                cmin=cmin,
            ),
            **kwargs,
        )


def plot_meshes(
    meshes: Sequence[Union[Mesh, Data]],
    titles: Sequence[str] = None,
    showtriangles: Sequence[bool] = None,
    showscales: Sequence[bool] = None,
    autoshow: bool = False,
    showlegend: bool = False,
    colorscales: Sequence[str] = None,
    reversescales: Sequence[bool] = None,
    cmax: Sequence[float] = None,
    cmin: Sequence[float] = None,
    **kwargs,
) -> Figure:
    """
    Plots multiple shapes
    :param meshes: a list of shapes to plot
    :param titles: a list of titles for each subplot
    :param showscales: whether to show the scale for each shape
    :param autoshow: if True show the Figure automatically
    :return: the Figure
    """
    myscene = dict(
        camera=dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-0.25, y=0.25, z=2.75),
        ),
        aspectmode="data",
    )
    fig = make_subplots(
        rows=1,
        cols=len(meshes),
        specs=[[{"is_3d": True}] * len(meshes)],
        subplot_titles=titles,
        horizontal_spacing=0,
        vertical_spacing=0,
    )

    for i, mesh in enumerate(meshes):
        mesh = _as_mesh(mesh)
        fig.add_trace(
            _plot_mesh(
                mesh,
                showtriangles=showtriangles[i] if showtriangles is not None else True,
                showscale=showscales[i] if showscales is not None else None,
                scene=f"scene{i+1}",
                colorscale=colorscales[i] if colorscales is not None else None,
                reversescale=reversescales[i] if reversescales is not None else None,
                cmax=cmax[i] if cmax is not None else None,
                cmin=cmin[i] if cmin is not None else None,
                **kwargs,
            ),
            row=1,
            col=i + 1,
        )
    for i in range(len(meshes)):
        fig["layout"][f"scene{i+1}"].update(myscene)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))

    if autoshow:
        fig.show()

    fig.update_layout(showlegend=showlegend)

    return fig


def plot_mesh(
    mesh: Union[Mesh, Data, str],
    showtriangles: bool = True,
    showscale: bool = False,
    autoshow: bool = False,
    colorscale="Viridis",
    reversescale=False,
    cmax: float = None,
    cmin: float = None,
    **kwargs,
) -> Figure:
    """
    Plots multiple shapes
    :param mesh: a shape to plot
    :param autoshow: if True show the Figure automatically
    :return: the Figure
    """
    if isinstance(mesh, str) or isinstance(mesh, Path):
        mesh = meshio.read(mesh)
    _as_mesh(mesh)

    fig = plot_meshes(
        meshes=[mesh],
        showtriangles=[showtriangles],
        showscales=[showscale],
        autoshow=autoshow,
        colorscales=[colorscale],
        reversescales=[reversescale],
        cmax=[cmax],
        cmin=[cmin],
        **kwargs,
    )

    return fig


def add_points(
    fig: Figure,
    vertices: Union[Tuple[np.ndarray], np.ndarray],
    color: Union[str, np.ndarray] = "black",
    size=10,
):
    if isinstance(vertices, np.ndarray) or isinstance(vertices, torch.Tensor):
        vertices = (
            vertices[..., 0].squeeze().numpy(),
            vertices[..., 1].squeeze().numpy(),
            vertices[..., 2].squeeze().numpy(),
        )
    x, y, z = vertices

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                colorscale="Viridis",
                opacity=1,
            ),
        ),
        row=1,
        col=1,
    )
    fig.update_layout(showlegend=False)


def convert_colors(colors):
    converted_colors = colors
    if colors.shape[-1] == 3:
        converted_colors = []
        for i in range(colors.shape[0]):
            c = (colors[i]).astype(np.int32).tolist()
            converted_colors.append(f"rgb({c[0]}, {c[1]}, {c[2]})")
    return converted_colors


def get_point_colors(points, frequency=np.pi, rgb_rescale=True):
    colors = (points - points.min(axis=0)[0]) / (
        points.max(axis=0)[0] - points.min(axis=0)[0]
    )
    colors = np.cos(frequency * colors)
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    if rgb_rescale:
        colors *= 255
    return colors


def plot3dmesh(vertices: np.ndarray, faces: np.ndarray, color) -> None:
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=1,
                intensity=color,
                showscale=False,
            )
        ]
    )
    fig.show()


def get_cumulative_geo_errors(geo_errors):
    x = np.linspace(0, 1, 1000)
    y = np.mean(geo_errors[:, None] <= x[None, :], 0)
    return x, y


def plot_geo_errors(geo_errors):
    """
    Cumulative geodesic error plot.
        - On the x-axis the geodesic error
        - On the y-axis the portion of points that have less than that geo error
    Args:
        geo_errors:

    Returns:

    """
    x = np.linspace(0, 1, 1000)
    y = np.mean(geo_errors[:, None] <= x[None, :], 0)

    fig, ax = plt.subplots()
    ax.set_ylabel("% vertices within the error")
    ax.set_xlabel("geodesic error")
    ax.plot(x, y)
    ax.grid()
    return fig


if __name__ == "__main__":
    keops_cdist_argmin(torch.randn(1000, 2), torch.randn(2000, 2), 0)
