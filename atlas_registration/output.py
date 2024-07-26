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
"""distribution of registration results as mesoscaler-format HDF5 files"""

from typing import Optional, Dict, Iterable, Tuple, Generator
from pathlib import Path
from collections import namedtuple as _namedtuple
import json as _json

import numpy as _np
import h5py as _h5

import affine2d as _affine
import affinealigner as _aa
import mesoscaler as _meso

from .types import (
    PathLike,
)
from . import (
    rois as _rois,
)


class AnimalAtlasRegistration(_namedtuple('AnimalAtlas', ('batch', 'animal', 'image', 'from_ref'))):
    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]


class MultiSessionAlignment(_namedtuple('SessionAtlas', ('batch', 'animal', 'date', 'type', 'image', 'aligned', 'from_animal'))):
    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def name(self) -> str:
        if self.type == 'task':
            suffix = ''
        else:
            suffix = f"_{self.type}"
        return f"{self.date}_{self.animal}{suffix}"

    @property
    def longtype(self) -> str:
        if self.type == 'ss':
            return 'sensory-stim'
        elif self.type == 'rest':
            return 'resting-state'
        elif self.type == 'task':
            return 'task'
        else:
            raise ValueError(f"unexpected type: {self.type}")

    def metadata(self) -> Dict[str, str]:
        return {
            'batch': self.batch,
            'animal': self.animal,
            'date': self.date,
            'type': self.longtype,
        }


class StoredDataset(_namedtuple('StoredDataset', ('session', 'from_ref', 'rois'))):
    def list_roi_names(self) -> Tuple[str]:
        names = []
        for roi in self.rois:
            if roi.name not in names:
                names.append(roi.name)
        return tuple(names)

    def create_bordered_image(
        self,
        width: int = 2,
        color: _aa.types.ColorSpec = 'w',
        alpha: float = 1.0,
        rgb24: bool = True,
    ) -> _aa.types.RGB24Image:
        bordered = _rois.overlay_ROI_borders(
            self.session.image,
            self.rois,
            border_width=width,
            border_color=color,
            border_alpha=alpha,
        )
        if rgb24 == False:
            bordered = bordered.mean(2).astype(_np.uint8)
        return bordered

    def rois_in_uint8(self) -> Generator[_meso.rois.ROI, None, None]:
        for roi in self.rois:
            yield _meso.rois.ROI(
                name=roi.name,
                side=roi.side,
                AllenID=roi.AllenID,
                description=roi.description,
                mask=(roi.mask > 0).astype(_np.uint8) * 255,
            )


def load_batch_registration_file(
    batch: str,
    regfile: PathLike,
) -> Tuple[AnimalAtlasRegistration]:
    regs = []
    with _h5.File(regfile, 'r') as src:
        animals = _json.loads(src.attrs['animals'])
        for i, animal in enumerate(animals):
            animal_img = _np.array(src['aligned_images'][i, :, :])
            atlas_to_animal = _np.array(src['ref512_to_animal'][i, :, :])
            reg = AnimalAtlasRegistration(
                batch=batch,
                animal=animal,
                image=animal_img,
                from_ref=atlas_to_animal.astype(_np.float32)
            )
            regs.append(reg)
    return tuple(regs)


def load_animal_alignment_file(
    regfile: PathLike,
) -> Tuple[MultiSessionAlignment]:
    regs = []
    with _h5.File(regfile, 'r') as src:
        sessions = _json.loads(src.attrs['sessions'])
        aligned  = _np.array(src['mean/aligned']).mean(0)
        for i, sess in enumerate(sessions):
            session_img = _np.array(src['mean/original'][i, :, :])
            to_animal = _np.array(src['transform'][i, :, :])
            reg = MultiSessionAlignment(
                batch=sess['batch'],
                animal=sess['animal'],
                date=sess['date'],
                type=sess['type'],
                image=session_img,
                aligned=aligned,
                from_animal=_affine.invert(to_animal).astype(_np.float32)
            )
            regs.append(reg)
    return tuple(regs)


def prepare_data_to_store(
    session: MultiSessionAlignment,
    animal: AnimalAtlasRegistration,
    atlas: Optional[Iterable[_meso.rois.ROI]] = None,
) -> StoredDataset:
    if atlas is None:
        atlas = _rois.load_reference_ROIs()
    ref_to_sess = _affine.compose(animal.from_ref, session.from_animal).astype(_np.float32)
    rois = _rois.warp_ROIs(atlas, ref_to_sess, width=session.width, height=session.height)
    return StoredDataset(session=session, from_ref=ref_to_sess, rois=rois)


def write_dataset(
    outfile: PathLike,
    dataset: StoredDataset,
    compression: Optional[int] = 9,
):
    outfile = Path(outfile)
    if not outfile.parent.exists():
        outfile.parent.mkdir(parents=True)
    opts = dict(compression=compression)
    with _h5.File(outfile, 'w') as out:
        for key, val in dataset.session.metadata().items():
            out.attrs[key] = val
        # affine transformation matrices
        group = out.create_group('transform')
        group.attrs['description'] = "key affine transformation matrices for atlas registration"
        ent = group.create_dataset('atlas_to_data', data=dataset.from_ref)
        ent.attrs['description'] = "the affine matrix in the shape (2, 3) representing the transformation from the 512x512 atlas to the session coordinates (original scale)"
        ent = group.create_dataset('animalavg_to_data', data=dataset.session.from_animal)
        ent.attrs['description'] = "the affine matrix in the shape (2, 3) representing the transformation from the animal-average coordinates to the session coordinates (original scale)"
        # images
        group = out.create_group('images')
        group.attrs['description'] = "key images for visual validation of atlas registration"
        ent = group.create_dataset('source', data=dataset.session.image, **opts)
        ent.attrs['description'] = "the scaled mean-signal image of this session"
        ent = group.create_dataset('aligned', data=dataset.session.aligned, **opts)
        ent.attrs['description'] = "the mean-signal image of this session, being transformed into the animal-average coordinate space"
        ent = group.create_dataset('borders', data=dataset.create_bordered_image(rgb24=False), **opts)
        ent.attrs['description'] = "the atlas borders being overlaid on top of the scaled mean-signal image of this session"
        # ROIs
        group = out.create_group('rois')
        group.attrs['names'] = _json.dumps(dataset.list_roi_names())
        group.attrs['description'] = "the registered atlas ROI masks for this session"
        for roi in dataset.rois_in_uint8():
            roi._write_hdf(parent=group, **opts)

