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
"""a toolkit and a pipeline for mouse atlas registration"""
from importlib import reload as _reload  # DEBUG

from . import (
    types,
    defaults,
    alignment,
    registration,
    rois,
    output,
    batch,
)

# DEBUG
_reload(types)
_reload(defaults)
_reload(alignment)
_reload(registration)
_reload(rois)
_reload(output)
_reload(batch)

align_sessions = alignment.align_sessions
write_aligned_sessions = alignment.write_aligned_sessions

register_animal_average_frames = registration.register_animal_average_frames
write_atlas_registration = registration.write_atlas_registration

load_reference_ROIs = rois.load_reference_ROIs
warp_ROIs = rois.warp_ROIs
overlay_ROI_borders = rois.overlay_ROI_borders

load_batch_registration_file = output.load_batch_registration_file
load_animal_alignment_file = output.load_animal_alignment_file

align_sessions_for_animal = batch.align_sessions_for_animal
register_atlas_for_batch = batch.register_atlas_for_batch
export_registration_for_batch = batch.export_registration_for_batch
process_batch = batch.process_batch

