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

"""within-animal, across-session alignment procedures"""
from typing import Iterable, Optional
from typing_extensions import Self
from pathlib import Path
from collections import namedtuple as _namedtuple
import sys as _sys
import json as _json

import numpy as _np
import h5py as _h5

import affine2d as _affine
import affinealigner as _aa
from session_explorer import RawData

from .types import (
    PathLike,
)


class AlignedSessions(_namedtuple('AlignedSessions', ('sessions', 'mean', 'std', 'transform'))):
    def scaled(self) -> Self:
        mean_scaled = [_aa.compute.scale_image(img) for img in self.mean]
        std_scaled  = [_aa.compute.scale_image(img) for img in self.std]
        return self._replace(mean=mean_scaled, std=std_scaled)


def align_sessions(
    sessions: Iterable[RawData],
) -> AlignedSessions:
    sessions = tuple(sessions)
    meanimgs = []
    stdimgs = []
    readsessions = []
    for sess in sessions:
        try:
            m, s = sess.read_avg_frames()
            meanimgs.append(m)
            stdimgs.append(s)
            readsessions.append(sess)
        except BaseException as e:
            print(f"***{sess.base}: {e}", flush=True, file=_sys.stderr)
    transforms = _aa.align_images(meanimgs)
    return AlignedSessions(
        sessions=tuple(readsessions),
        mean=tuple(meanimgs),
        std=tuple(stdimgs),
        transform=tuple(transforms),
    )


def write_aligned_sessions(
    outfile: PathLike,
    aligned: AlignedSessions,
    compression: Optional[int] = 9,
):
    aligned = aligned.scaled()
    imgs = dict(mean=dict(), std=dict())
    imgs['mean']['original'] = _np.stack(aligned.mean, axis=0)
    imgs['std']['original']  = _np.stack(aligned.std, axis=0)
    imgs['mean']['aligned'] = _np.stack([_affine.warp_image(img, trans) for img, trans in zip(aligned.mean, aligned.transform)], axis=0)
    imgs['std']['aligned']  = _np.stack([_affine.warp_image(img, trans) for img, trans in zip(aligned.std, aligned.transform)], axis=0)
    transform = _np.stack(aligned.transform, axis=0)
    H, W = aligned.mean[0].shape  # assumed to be a 2-D image

    outfile = Path(outfile)
    if not outfile.parent.exists():
        outfile.parent.mkdir(parents=True)
    opts = dict(compression=compression)
    with _h5.File(outfile, 'w') as out:
        out.attrs['sessions'] = _json.dumps([sess.metadata() for sess in aligned.sessions])
        out.attrs['image_width'] = W
        out.attrs['image_height'] = H
        ent_t = out.create_dataset('transform', data=transform, **opts)
        ent_t.attrs['description'] = "Affine transform matrices in the shape (N, 2, 3), representing transforms from individual sessions to the per-animal template."
        for imgtype, formats in imgs.items():
            group = out.create_group(imgtype)
            for aligntype, images in formats.items():
                ent_i = group.create_dataset(aligntype, data=images, chunks=(1, H, W), **opts)
                ent_i.attrs['description'] = f"the {imgtype} images ({aligntype}), concatenated into the shape (N, H, W)."
