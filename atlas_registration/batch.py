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
from typing import Iterable, Optional
from pathlib import Path
import json as _json

import h5py as _h5
from tqdm import tqdm as _tqdm

import bdbc_session_explorer as _sessx

from .types import (
    PathLike,
    PathsLike,
)
from . import (
    defaults as _defaults,
    alignment as _alignment,
    registration as _registration,
    rois as _rois,
    output as _output,
)


def process_batch(
    batch: str,
    sessions_root: Optional[PathLike] = None,
    rawdata_root: Optional[PathsLike] = None,
    registration_root: Optional[PathLike] = None,
    rawdata_version: _sessx.rawdata.RawFileVersion = 'v2',
    rawdata_errors: _sessx.core.ErrorHandling = 'ignore',
    hdf_compression: Optional[int] = 9,
    hclip: Optional[slice] = _defaults.REGISTRATION_HCLIP,
    verbose: bool = True,
):
    def _log(msg, end='\n'):
        if verbose == True:
            print(msg, end=end, flush=True)

    # collect sessions by animal
    # FIXME: here it does not have to be Rawdata instances
    # (can be Session instances instead)
    sessions: dict[str, list[_sessx.RawData]] = dict()
    for sess in _sessx.iterate_sessions(
        batch=batch,
        sessions_root_dir=sessions_root,
        verbose=verbose,
    ):
        if not sess.has_rawdata():
            continue
        rawdata = _sessx.rawdata_from_session(
            sess,
            rawroot=rawdata_root,
            file_version=rawdata_version,
            error_handling=rawdata_errors,
        )
        if rawdata.path is None:
            raise FileNotFoundError(f'failed to locate rawdata for: {sess.shortbase}')
        if sess.animal not in sessions.keys():
            sessions[sess.animal] = []
        sessions[sess.animal].append(rawdata)
    if len(sessions) == 0:
        raise ValueError('no sessions found')
    _count_batch(sessions, verbose=verbose)

    alignments = dict()
    for animal, rawfiles in sessions.items():
        _log(f"align: {animal}...", end=' ')
        alignments[animal] = align_sessions_for_animal(
            rootdir=registration_root,
            animal_rawdata_files=rawfiles,
            batch=batch,
            animal=animal,
            compression=hdf_compression,
        )
        _log("done.")
    register_atlas_for_batch(
        rootdir=registration_root,
        alignfiles=alignments,
        batch=batch,
        hclip=hclip,
        compression=hdf_compression,
    )
    export_registration_for_batch(
        rootdir=registration_root,
        batch=batch,
        compression=hdf_compression,
        verbose=verbose
    )


def _count_batch(
    sessions: dict[str, Iterable[_sessx.RawData]],
    verbose: bool = True
):
    if verbose == False:
        return
    print(f"found {len(sessions)} animals:", end=' ')
    indiv = []
    for animal, sessx in sessions.items():
        sessx = tuple(sessx)
        indiv.append(f"{animal} ({len(sessx)})")
    print(', '.join(indiv), flush=True)


def align_sessions_for_animal(
    rootdir: PathLike,
    animal_rawdata_files: Iterable[_sessx.RawData],
    batch: Optional[str] = None,
    animal: Optional[str] = None,
    compression: Optional[int] = 9,
) -> Path:
    rawdata_set = tuple(animal_rawdata_files)
    if batch is None:
        batch = rawdata_set[0].session.batch
    if animal is None:
        animal = rawdata_set[0].session.animal
    aligned = _alignment.align_sessions(rawdata_set)
    outpath = _defaults.animal_alignment_file(rootdir, batch=batch, animal=animal)
    _alignment.write_aligned_sessions(outpath, aligned, compression=compression)
    return outpath


def register_atlas_for_batch(
    rootdir: PathLike,
    alignfiles: dict[str, PathLike],
    batch: Optional[str] = None,
    hclip: Optional[slice] = _defaults.REGISTRATION_HCLIP,
    compression: Optional[int] = 9,
) -> Path:
    if batch is None:
        batch = _estimate_batch(alignfiles)
    reg = _registration.register_animal_average_frames(alignfiles, hclip=hclip)
    outpath = _defaults.atlas_registration_file(rootdir, batch=batch)
    _registration.write_atlas_registration(outpath, reg, compression=compression)
    return outpath


def _estimate_batch(alignfiles: dict[str, PathLike]) -> str:
    paths = tuple(alignfiles.values())
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
    atlas = _rois.load_reference_ROIs()
    batchfile = _defaults.atlas_registration_file(rootdir, batch)
    for animal_reg in _output.load_batch_registration_file(batch, batchfile):
        animalfile = _defaults.animal_alignment_file(rootdir, batch, animal_reg.animal)
        animaldir = animalfile.parent

        session_regs: tuple[_output.SessionAlignment] = _output.load_animal_alignment_file(animalfile)
        if verbose == True:
            session_regs = _tqdm(session_regs, desc=animal_reg.animal)
        for session_reg in session_regs:
            outpath = animaldir / f"{session_reg.name}_mesoscaler.h5"
            data: _output.StoredDataset = _output.prepare_data_to_store(session_reg, animal_reg, atlas=atlas)
            _output.write_dataset(outpath, data, compression=compression)
