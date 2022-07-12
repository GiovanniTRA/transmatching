import numpy as np
import streamlit as st
from pytorch_lightning import seed_everything
from stqdm import stqdm

from evaluation.competitors.eval_dataset import EvalDataset
from evaluation.utils import (
    PROJECT_ROOT,
    Mesh,
    convert_colors,
    get_dists,
    get_hydra_cfg,
    get_point_colors,
    plot_meshes,
)

st.markdown("App to generate a nice point-wise color for each evaluation dataset.")


seed_everything(0)

datasets = [
    x.name for x in (PROJECT_ROOT / "evaluation" / "datasets").iterdir() if x.is_dir()
]
dataset_name = st.selectbox("Select dataset to consider:", datasets)
dataset = EvalDataset(dataset_name)

sample_idx = st.number_input("Select sample index", min_value=0, max_value=len(dataset))
sample = dataset[sample_idx]

geo_dists = get_dists()
cfg = get_hydra_cfg()


points_A = sample["points_A"]

frequency = st.number_input("Select color frequency:", value=np.pi)
color = get_point_colors(
    sample["points_A"],
    frequency=frequency,
)

f = plot_meshes(
    meshes=[
        Mesh(
            v=points_A,
            f=None,
            color=convert_colors(color),
        ),
    ],
    titles=["Shape A"],
    showtriangles=[False],
    showscales=None,
    autoshow=False,
)
st.plotly_chart(f, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    st.subheader("Dataset wise color")
    st.markdown(
        "Export color for the whole dataset. "
        "The **shapes in the dataset must be in correspondence**!"
    )
    if st.button("Export global color"):
        colorsfile = (
            PROJECT_ROOT
            / "evaluation"
            / "datasets"
            / dataset_name
            / "data"
            / "colors.npy"
        )
        np.save(colorsfile, color)
        st.info(f"Saved: `{colorsfile}`")

with col2:
    st.subheader("Shape wise color")
    st.markdown(
        "Export color for each shape in each sample of the dataset. "
        "The **coloring may change between shapes**!"
    )
    if st.button("Export all colors"):
        for sample in stqdm(dataset):
            path = sample["path"]
            for shape, file in [
                ("points_A", "colors_A.npy"),
                ("points_B", "colors_B.npy"),
            ]:
                colorsfile = path / file
                color = get_point_colors(
                    sample[shape],
                    frequency=frequency,
                )
                np.save(colorsfile, color)
        st.info(
            f"Saved `{len(dataset)* 2}` colors under <`{path.parent}/*/color_*.npy`>"
        )
# dataset = get_dataset(cfg)
#
# colorsfile = Path(get_env("SURREAL_COLOR"))
#
# set_dataset_augmentations(dataset)
#
#
# st.button("Rerun")
# sampledidx = st.number_input("Sample idx", min_value=0, max_value=len(dataset), value=6)
# sample = dataset[sampledidx]
#
# shape_A = sample["shape_A"]
# shape_B = sample["shape_B"]
# permuted_shape_A = sample["permuted_shape_A"]
# permuted_shape_B = sample["permuted_shape_B"]
# permutations_from_A_to_B = sample["permutation_from_A_to_B"]
#
#
# s = permuted_shape_A
#
# # color = get_point_colors(shape_A, frequency=st.number_input("Freq:", value=np.pi))
# color = np.load(Path(get_env("SURREAL_COLOR")))
#
# plot_color = color[sample["permutation_A"]]
# f = plot_meshes(
#     meshes=[
#         Mesh(
#             v=s,
#             f=None,
#             color=convert_colors(plot_color),
#         ),
#     ],
#     titles=["Shape A"],
#     showtriangles=[False],
#     showscales=None,
#     autoshow=False,
# )
# st.plotly_chart(f, use_container_width=True)
#
# st.write(color)
#
#
