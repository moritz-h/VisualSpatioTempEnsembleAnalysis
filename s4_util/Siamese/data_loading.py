import glob
import os

import scipy.ndimage
import numpy as np

from s4_util.PythonExtras import volume_tools, numpy_extras as npe
from s4_util.Siamese.data_types import *


class EnsembleMemberMetadata(NamedTuple):
    memberNames: List[str]
    memberPaths: List[str]
    memberShapes: List[TupleInt]
    memberShapesOrig: List[TupleInt]

    def get_shape_min_max(self):
        return self._get_shape_min_max(self.memberShapes)

    def get_orig_shape_min_max(self):
        return self._get_shape_min_max(self.memberShapesOrig)

    def get_attr_number(self) -> int:
        assert all(len(s) == SHAPE_LEN for s in self.memberShapes)
        assert all(s[-1] == self.memberShapes[0][-1] for s in self.memberShapes)

        return self.memberShapes[0][-1]

    def is_2d(self) -> bool:
        return all(s[1] == 1 for s in self.memberShapes)  # All z dimensions have size one.

    def is_single_member(self) -> bool:
        return len(self.memberNames) == 1

    @staticmethod
    def _get_shape_min_max(shapes: List[TupleInt]) -> Tuple[TupleInt, TupleInt]:
        allShapesArray = np.asarray(shapes)

        shapeMin = tuple(np.min(allShapesArray, axis=0))
        shapeMax = tuple(np.max(allShapesArray, axis=0))

        return shapeMin, shapeMax


def load_ensemble_member_metadata(dataPath: str,
                                  downsampleFactor: int,
                                  volumeCrop: Tuple[slice, ...]) -> EnsembleMemberMetadata:
    volumeCrop = (SHAPE_LEN_NA * (slice(None),)) if not volumeCrop else volumeCrop

    assert len(volumeCrop) == SHAPE_LEN_NA
    volumeCropWithAttr = volumeCrop + (slice(None),)  # Take all the attributes.

    memberPaths = glob.glob(dataPath)
    memberNames = []
    memberShapes = []
    memberShapesOrig = []
    
    if len(memberPaths) == 0:
        raise ValueError("Data doesn't exist at '{}'.".format(dataPath))

    for metaPath in memberPaths:
        meta = volume_tools.VolumeMetadata.load_from_dat(metaPath)
        name = os.path.splitext(os.path.basename(metaPath))[0]
        shape = meta.get_shape(forceMultivar=True)  # type: TupleInt
        assert len(shape) == SHAPE_LEN
        memberShapesOrig.append(shape)

        is3d = shape[1] > 1
        scaleFactors = (1, downsampleFactor if is3d else 1, downsampleFactor, downsampleFactor, 1)
        shape = volume_tools.downsample_shape(shape, volumeCropWithAttr, scaleFactors)

        memberShapes.append(shape)
        memberNames.append(name)

    return EnsembleMemberMetadata(memberNames, memberPaths, memberShapes, memberShapesOrig)


def downsample_volume(volumeData: npe.LargeArray, factor: float) -> npe.LargeArray:
    assert isinstance(volumeData, np.ndarray)  # For now, we assume in-memory numpy arrays.

    is3d = volumeData.shape[1] > 1
    zoomFactors = [1, 1 / factor if is3d else 1, 1 / factor, 1 / factor]
    attrAxis = SHAPE_LEN - 1

    # We downsample different attribute fields separately to avoid mixing attributes with bilinear interpolation.
    volumesByAttribute = []
    for i in range(volumeData.shape[attrAxis]):
        volumesByAttribute.append(
            scipy.ndimage.interpolation.zoom(volumeData[..., i], zoomFactors, order=1, mode='nearest')
        )

    return np.stack(volumesByAttribute, axis=attrAxis)
