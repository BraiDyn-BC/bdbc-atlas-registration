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
"""batch processing functions"""
from typing import Iterable, Optional, Dict
from pathlib import Path
import json as _json

import h5py as _h5
from tqdm import tqdm as _tqdm

from rawdatareader import RawData

from .types import (
    PathLike,
)
from . import (
    defaults as _defaults,
    alignment as _alignment,
    registration as _registration,
    rois as _rois,
    output as _output,
)


def align_sessions_for_animal(
    rootdir: PathLike,
    animal_sessions: Iterable[RawData],
    batch: Optional[str] = None,
    animal: Optional[str] = None,
    compression: Optional[int] = 9,
) -> Path:
    sessions = tuple(animal_sessions)
    if batch is None:
        batch = sessions[0].batch
    if animal is None:
        animal = sessions[0].animal
    aligned = _alignment.align_sessions(sessions)
    outpath = _defaults.animal_alignment_file(rootdir, batch=batch, animal=animal)
    _alignment.write_aligned_sessions(outpath, aligned, compression=compression)
    return outpath


def register_atlas_for_batch(
    rootdir: PathLike,
    animalpaths: Dict[str, PathLike],
    batch: Optional[str] = None,
    hclip: Optional[slice] = slice(None, 48),
    compression: Optional[int] = 9,
) -> Path:
    if batch is None:
        batch = _estimate_batch(animalpaths)
    rootdir = Path(rootdir)
    reg = _registration.register_animal_average_frames(animalpaths, hclip=hclip)
    outpath = _defaults.atlas_registration_file(rootdir, batch=batch)
    _registration.write_atlas_registration(outpath, reg, compression=compression)
    return outpath


def _estimate_batch(animalpaths: Dict[str, PathLike]) -> str:
    paths = tuple(animalpaths.values())
    if len(paths) == 0:
        raise ValueError('no animals found')
    with _h5.File(paths[0], 'r') as src:
        sessions = _json.loads(src.attrs['sessions'])
    return sessions[0]['batch']


def export_registration_for_batch(
    rootdir: PathLike,
    batch: str,
    compression: Optional[int] = 9,
    verbose: bool = True,
):
    rootdir = Path(rootdir)
    atlas = _rois.load_reference_ROIs()
    batchfile = rootdir / batch / 'ATLAS-REG.h5'
    for animal_reg in _output.load_batch_registration_file(batch, batchfile):
        animaldir = rootdir / animal_reg.batch / animal_reg.animal
        animalfile = animaldir / f"{animal_reg.animal}_ALIGNED.h5"

        session_regs = _output.load_animal_alignment_file(animalfile)
        if verbose == True:
            session_regs = _tqdm(session_regs, desc=animal_reg.animal)
        for session_reg in session_regs:
            outpath = animaldir / f"{session_reg.name}_mesoscaler.h5"
            data = _output.prepare_data_to_store(session_reg, animal_reg, atlas=atlas)
            _output.write_dataset(outpath, data, compression=compression)
