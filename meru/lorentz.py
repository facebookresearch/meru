# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.
This model represents a hyperbolic space of `d` dimensions on the upper-half of
a two-sheeted hyperboloid in a Euclidean space of `(d+1)` dimensions.

Hyperbolic geometry has a direct connection to the study of special relativity
theory -- implementations in this module borrow some of its terminology. The axis
of symmetry of the Hyperboloid is called the _time dimension_, while all other
axes are collectively called _space dimensions_.

All functions implemented here only input/output the space components, while
while calculating the time component according to the Hyperboloid constraint:

    `x_time = torch.sqrt(1 / curv + torch.norm(x_space) ** 2)`
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise Lorentzian inner product
        between input vectors.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the hyperboloid.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the hyperboloid.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving space components of the mapped
        vectors on the hyperboloid.
    """

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    # Ensure numerical stability in sinh by clamping input.
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as `x`, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """

    # Calculate distance of vectors to the hyperboloid vertex.
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    """
    Compute the half aperture angle of the entailment cone formed by vectors on
    the hyperboloid. The given vector would meet the apex of this cone, and the
    cone itself extends outwards to infinity.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        min_radius: Radius of a small neighborhood around vertex of the hyperboloid
            where cone aperture is left undefined. Input vectors lying inside this
            neighborhood (having smaller norm) will be projected on the boundary.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving the half-aperture of entailment cones
        formed by input vectors. Values of this tensor lie in `(0, pi/2)`.
    """

    # Ensure numerical stability in arc-sin by clamping input.
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):
    """
    Given two vectors `x` and `y` on the hyperboloid, compute the exterior
    angle at `x` in the hyperbolic triangle `Oxy` where `O` is the origin
    of the hyperboloid.

    This expression is derived using the Hyperbolic law of cosines.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        y: Tensor of same shape as `x` giving another batch of vectors.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(B, )` giving the required angle. Values of this
        tensor lie in `(0, pi)`.
    """

    # Calculate time components of inputs (multiplied with `sqrt(curv)`):
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    # Calculate lorentzian inner product multiplied with curvature. We do not use
    # the `pairwise_inner` implementation to save some operations (since we only
    # need the diagonal elements).
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    # Make the numerator and denominator for input to arc-cosh, shape: (B, )
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle
