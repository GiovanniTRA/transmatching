import numpy as np
from transmatching.Utils.utils import est_area

from evaluation.utils import calc_tri_areas


def unit_area_normalization(points, faces):
    area_A = np.sqrt(calc_tri_areas(points, faces).sum())
    points = points / area_A
    points -= points.mean(0)
    return points


def naive_normalization(points, rescale: bool = True):
    if rescale:
        points = points * 0.741
    points = points - points.mean(0)
    return points


def area_weighted_denormalization(points, reference_points, rescale: bool = True):
    if rescale:
        reference_points = reference_points * 0.741

    shape_area = est_area(reference_points[None, ...])[0]
    points = points + (
        reference_points * (shape_area / shape_area.sum(-1, keepdims=True))[..., None]
    ).sum(-2, keepdims=True)

    if rescale:
        points = points * (1 / 0.741)

    return points


def area_weighted_normalization(shape, rescale: bool = True):
    if rescale:
        shape = shape * 0.741

    shape_area = est_area(shape[None, ...])[0]
    shape = shape - (
        shape * (shape_area / shape_area.sum(-1, keepdims=True))[..., None]
    ).sum(-2, keepdims=True)
    return shape


def normalization_wrt_template_area(shape, faces, template_area):
    shape_area = np.sqrt(calc_tri_areas(shape, faces).sum())
    shape = shape * (template_area / shape_area)
    shape -= shape.mean(0)
    return shape


def normalization_wrt_lowres_mesh(faust_shape, faust_1k):
    """
    It was used for faust and faust_permuted
    Args:
        faust_shape:
        faust_1k:

    Returns:

    """
    S = (
        (faust_1k.max(0) - faust_1k.min(0)) / (faust_shape.max(0) - faust_shape.min(0))
    )[1]
    faust_shape = faust_shape * S

    T = ((faust_shape.min(0) + faust_shape.max(0)) / 2) - (
        (faust_1k.min(0) + faust_1k.max(0)) / 2
    )
    faust_shape = faust_shape - T

    return faust_shape
