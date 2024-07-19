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
"""animal-by-animal registration to Allen CCF atlas using MesoNet"""

from typing import Dict, Optional
from pathlib import Path
from collections import namedtuple as _namedtuple
import json as _json
import tempfile as _tempfile
import shutil as _shutil

import numpy as _np
import h5py as _h5
import pandas as _pd
import imageio.v3 as _iio
import cv2 as _cv2

import affine2d as _affine
import affinealigner as _aa
import mesoscaler as _meso

from .types import (
    PathLike,
)


class AtlasRegistration(_namedtuple('AtlasRegistration', ('metadata', 'images', 'landmarks'))):
    @property
    def num_images(self) -> int:
        return self.images.shape[0]

    @property
    def width(self) -> int:
        return self.images.shape[2]

    @property
    def height(self) -> int:
        return self.images.shape[1]


def register_animal_average_frames(
    animalpaths: Dict[str, PathLike],
    hclip: Optional[slice] = slice(None, 48),
) -> AtlasRegistration:
    workdir = Path(_tempfile.mkdtemp(prefix='mesoscaler'))
    imgdir = workdir / 'images'
    imgdir.mkdir()
    try:
        for animal, regfile in animalpaths.items():
            with _h5.File(regfile, 'r') as src:
                stdimg = _np.array(src['std/aligned'], copy=False).mean(0)
            stdimg = _aa.compute.to_uint8(stdimg)
            if hclip is not None:
                stdimg[hclip, :] = 0
            _iio.imwrite(str(imgdir / f"{animal}.png"), stdimg)
        # mesoscaler
        collectdir = _meso.procs.run_image_collection(imgdir, workdir / 'collected')
        landmarkdir = _meso.procs.run_landmark_prediction(collectdir, workdir / 'landmarks')
        aligndir = _meso.procs.run_landmark_alignment(landmarkdir, workdir / 'alignment', separate_sides=False)
        # read results
        metadata = _pd.read_csv(str(collectdir / 'metadata.csv')).drop(['Frame', 'TotalFrames'], axis=1)
        metadata['Image'] = [str(val).replace('.png', '') for val in metadata.Image.values]
        metadata.columns = ('Animal', 'Width', 'Height')
        alignment = _pd.read_csv(str(aligndir / 'reference_to_images_transform.csv')).drop(
            ['is_separate', 'right_xx', 'right_xy', 'right_xc', 'right_yx', 'right_yy', 'right_yc'], axis=1
        )
        alignment.columns = ('xx', 'xy', 'xc', 'yx', 'yy', 'yc')
        metadata = _pd.concat([metadata, alignment], axis=1)
        alignimgs = _iio.imread(aligndir / 'images_with_aligned_landmarks.mp4')
        landmarks = _pd.read_csv(str(landmarkdir / 'landmarks.csv'), header=[0, 1, 2]).droplevel(0, axis=1)
        return AtlasRegistration(metadata, alignimgs, landmarks)
    finally:
        _shutil.rmtree(workdir)


def write_atlas_registration(
    outfile: PathLike,
    reg: AtlasRegistration,
    compression: Optional[int] = 9,
):
    outfile = Path(outfile)
    if not outfile.parent.exists():
        outfile.parent.mkdir(parents=True)

    # prepare data in its original dims
    aligned_animals = []
    ref512_to_animal512 = []
    ref512_to_animal = []
    aligned_images = []
    rW = reg.width
    rH = reg.height
    oW = reg.metadata.iloc[0].Width
    oH = reg.metadata.iloc[0].Height
    scaler = _affine.scaler(oW / rW)
    for i, row in reg.metadata.iterrows():
        to_animal512 = _np.array([[row.xx, row.xy, row.xc], [row.yx, row.yy, row.yc]])
        to_animal = _affine.compose(to_animal512, scaler)
        img = _cv2.resize(reg.images[i], (oW, oH), interpolation=_cv2.INTER_AREA)
        aligned_animals.append(row.Animal)
        ref512_to_animal512.append(to_animal512)
        ref512_to_animal.append(to_animal)
        aligned_images.append(img)

    # write out
    opts = dict(compression=compression)
    with _h5.File(outfile, 'w') as out:
        out.attrs['animals'] = _json.dumps(aligned_animals)
        out.attrs['image_width'] = oW
        out.attrs['image_height'] = oH
        out.attrs['atlas_width'] = rW
        out.attrs['atlas_height'] = rH
        ent = out.create_dataset('ref512_to_animal512', data=_np.stack(ref512_to_animal512, axis=0), **opts)
        ent.attrs['description'] = f"Affine transform matrices in the shape (N, 2, 3), representing the conersion from the reference atlas (H{rH}xW{rW}px) to the data (scaled to H{rH}xW{rW}px)"
        ent = out.create_dataset('ref512_to_animal', data=_np.stack(ref512_to_animal, axis=0), **opts)
        ent.attrs['description'] = f"Affine transform matrices in the shape (N, 2, 3), representing the conersion from the reference atlas (H{rH}xW{rW}px) to the data (original H{oH}xW{oW}px)"
        ent = out.create_dataset('aligned_images', data=_np.stack(aligned_images, axis=0), **opts)
        ent.attrs['description'] = f"Aligned atlas landmarks overlaied on top of images, in the shape (N, {oH}, {oW})"
        landmarks = out.create_group('landmarks')
        landmarks.attrs['description'] = 'the estimated coordinates of the landmarks in the 512x512 space'
        names = sorted(set(col[0] for col in reg.landmarks.columns))
        for name in names:
            pt = landmarks.create_group(name)
            for ax in ('x', 'y', 'likelihood'):
                pt.create_dataset(ax, data=reg.landmarks[name, ax].values, **opts)

