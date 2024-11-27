# MIT License
#
# Copyright (c) 2024 Keisuke Sehara
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""procedures related to handling of ROIs"""

from typing import Iterable, Optional

import numpy as _np

import affine2d as _affine
import affinealigner as _aa
import mesoscaler as _meso


def load_reference_ROIs() -> tuple[_meso.rois.ROI]:
    outlines = _meso.rois.load_reference_outlines().rois
    atlas = _meso.rois.load_reference_ROIs().rois

    merged = _meso.rois.ROI(
        name='outline', side='both', AllenID=-1,
        description='the expected outline of the brain for this ROI set',
        mask=(outlines[0].mask.astype(_np.uint8) + outlines[1].mask.astype(_np.uint8))
    )
    return (merged,) + outlines + atlas


def warp_ROIs(
    rois: Iterable[_meso.rois.ROI],
    transform: _affine.AffineMatrix,
    width: int,
    height: Optional[int] = None,
) -> tuple[_meso.rois.ROI]:
    if height is None:
        height = width
    warped = []
    for roi in rois:
        mask = _affine.warp_image(roi.mask.astype(_np.uint8), transform, width=width, height=height) > 0
        warped.append(
            _meso.rois.ROI(
                name=roi.name,
                side=roi.side,
                AllenID=roi.AllenID,
                description=roi.description,
                mask=mask,
            )
        )
    return tuple(warped)


def overlay_ROI_borders(
    base_image: _aa.types.Image,
    rois: Iterable[_meso.rois.ROI],
    exclude_names: Iterable[str] = ('outline',),
    border_width: int = 2,
    border_color: _aa.types.ColorSpec = 'w',
    border_alpha: float = 1.0,
) -> _aa.types.RGB24Image:
    base_image = _aa.compute.to_rgb24(base_image)
    borders = []
    for roi in rois:
        if roi.name in exclude_names:
            continue
        border = _aa.compute.mask_to_border(roi.mask, border_width=border_width)
        borders.append(_aa.compute.color_mask(border, color=border_color, alpha=border_alpha))
    return _aa.compute.overlay(base_image, *borders)
