from typing import Optional, Union, Iterable, Dict, Tuple, Generator
from typing_extensions import Self
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
from tqdm import tqdm as _tqdm

import mesoscaler as _meso
import rawdatareader as _rr
import affinealigner as _aa
import affine2d as _affine

PathLike = Union[str, Path]


class AlignedSessions(_namedtuple('AlignedSessions', ('sessions', 'mean', 'std', 'transform'))):
    def scaled(self) -> Self:
        mean_scaled = [_aa.compute.scale_image(img) for img in self.mean]
        std_scaled  = [_aa.compute.scale_image(img) for img in self.std]
        return self._replace(mean=mean_scaled, std=std_scaled)


def align_sessions(
    sessions: Iterable[_rr.RawData],
) -> AlignedSessions:
    sessions = tuple(sessions)
    meanimgs = []
    stdimgs = []
    for sess in sessions:
        m, s = sess.read_avg_frames()
        meanimgs.append(m)
        stdimgs.append(s)
    transforms = _aa.align_images(meanimgs)
    return AlignedSessions(
        sessions=sessions,
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
    H, W = aligned.mean[0].shape  # assumes to be a 2-D image
    
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


def animal_session_alignment(
    rootdir: PathLike,
    animal_sessions: Iterable[_rr.RawData],
    batch: Optional[str] = None,
    animal: Optional[str] = None,
    compression: Optional[int] = 9,
) -> Path:
    rootdir = Path(rootdir)
    sessions = tuple(animal_sessions)
    if batch is None:
        batch = sessions[0].batch
    if animal is None:
        animal = sessions[0].animal
    aligned = align_sessions(sessions)
    outpath = rootdir / batch / animal / f"{animal}_ALIGNED.h5"
    write_aligned_sessions(outpath, aligned, compression=compression)
    return outpath


class AtlasRegistration(_namedtuple('AtlasRegistration', ('metadata', 'images'))):
    @property
    def num_images(self) -> int:
        return self.images.shape[0]

    @property
    def width(self) -> int:
        return self.images.shape[2]

    @property
    def height(self) -> int:
        return self.images.shape[1]


def register_animal_averages(
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
        return AtlasRegistration(metadata, alignimgs)
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
    opts = dict(compression = 9)
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


def batch_atlas_registration(
    rootdir: PathLike,
    animalpaths: Dict[str, PathLike],
    batch: Optional[str] = None,
    hclip: Optional[slice] = slice(None, 48),
    compression: Optional[int] = 9,
) -> Path:
    if batch is None:
        batch = _estimate_batch(animalpaths)
    rootdir = Path(rootdir)
    reg = register_animal_averages(animalpaths, hclip=hclip)
    outpath = rootdir / batch / 'ATLAS-REG.h5'
    write_atlas_registration(outpath, reg, compression=compression)
    return outpath


def _estimate_batch(animalpaths: Dict[str, PathLike]) -> str:
    paths = tuple(animalpaths.values())
    if len(paths) == 0:
        raise ValueError('no animals found')
    with _h5.File(paths[0], 'r') as src:
        sessions = _json.loads(src.attrs['sessions'])
    return sessions[0]['batch']


class AnimalAtlas(_namedtuple('AnimalAtlas', ('batch', 'animal', 'image', 'from_ref'))):
    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]


class SessionAtlas(_namedtuple('SessionAtlas', ('batch', 'animal', 'date', 'type', 'image', 'aligned', 'from_animal'))):
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


def load_batch_regs(
    batch: str,
    regfile: PathLike,
) -> Tuple[AnimalAtlas]:
    regs = []
    with _h5.File(regfile, 'r') as src:
        animals = _json.loads(src.attrs['animals'])
        for i, animal in enumerate(animals):
            animal_img = _np.array(src['aligned_images'][i, :, :])
            atlas_to_animal = _np.array(src['ref512_to_animal'][i, :, :])
            reg = AnimalAtlas(
                batch=batch,
                animal=animal,
                image=animal_img,
                from_ref=atlas_to_animal.astype(_np.float32)
            )
            regs.append(reg)
    return tuple(regs)


def load_animal_regs(
    regfile: PathLike,
) -> Tuple[SessionAtlas]:
    regs = []
    with _h5.File(regfile, 'r') as src:
        sessions = _json.loads(src.attrs['sessions'])
        aligned  = _np.array(src['mean/aligned']).mean(0)
        for i, sess in enumerate(sessions):
            session_img = _np.array(src['mean/original'][i, :, :])
            to_animal = _np.array(src['transform'][i, :, :])
            reg = SessionAtlas(
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


def load_reference_ROIs() -> Tuple[_meso.rois.ROI]:
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
) -> Tuple[_meso.rois.ROI]:
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
        bordered = overlay_ROI_borders(
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


def prepare_data_to_store(
    session: SessionAtlas,
    animal: AnimalAtlas,
    atlas: Optional[Iterable[_meso.rois.ROI]] = None,
) -> StoredDataset:
    if atlas is None:
        atlas = load_reference_ROIs()
    ref_to_sess = _affine.compose(animal.from_ref, session.from_animal).astype(_np.float32)
    rois = warp_ROIs(atlas, ref_to_sess, width=session.width, height=session.height)
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


def batch_export_registration(
    rootdir: PathLike,
    batch: str,
    compression: Optional[int] = 9,
    verbose: bool = True,
):
    rootdir = Path(rootdir)
    atlas = load_reference_ROIs()
    batchfile = rootdir / batch / 'ATLAS-REG.h5'
    for animal_reg in load_batch_regs(batch, batchfile):
        animaldir = rootdir / animal_reg.batch / animal_reg.animal
        animalfile = animaldir / f"{animal_reg.animal}_ALIGNED.h5"

        session_regs = load_animal_regs(animalfile)
        if verbose == True:
            session_regs = _tqdm(session_regs, desc=animal_reg.animal)
        for session_reg in load_animal_regs(animalfile):
            outpath = animaldir / f"{session_reg.name}_mesoscaler.h5"
            data = prepare_data_to_store(session_reg, animal_reg, atlas)
            write_dataset(outpath, data, compression=compression)
