import vital.data.config
from vital.data.config import DataTag
from vital.utils.parameters import parameters


class Label(DataTag):
    BG = 0
    ENDO = 1
    EPI = 2
    ATRIUM = 3


class View(DataTag):
    TWO = '2CH'
    FOUR = '4CH'


class Instant(DataTag):
    ED = 'ED'
    ES = 'ES'


@parameters
class DataTags:
    """Class to gather the tags referring to the different types of data stored in the HDF5 datasets.

    Args:
        registered: name of the tag indicating whether the dataset was registered.
        full_sequence: name of the tag indicating whether the dataset contains complete sequence between ED and ES for
                       each view.
        img_proc: name of the tag referring to resized images, used as input when training models.
        img: name of the tag referring to unprocessed images.
        gt_proc: name of the tag referring to resized groundtruths used as reference when training models.
        gt: name of the tag referring to unprocessed groundtruths, used as reference when evaluating models' scores.
    """
    registered: str = 'register'
    full_sequence: str = 'sequence'

    img_proc: str = 'img_proc'
    img: str = 'img'
    gt_proc: str = 'gt_proc'
    gt: str = 'gt'
    info: str = 'info'
    sequence_idx: str = 'seq_idx'


@parameters
class ResultTags(vital.data.config.ResultTags):
    """Class to gather the tags referring to CAMUS specific result data stored in the HDF5 result files.

    Args:
        info: name of the tag referring to images' metadata.
        proc_instants: name of the tag referring to metadata indicating which image where affected by the
                       postprocessing.
    """
    info: str = 'info'
    proc_instants: str = 'processed_instants'


image_size = 256
in_channels = 1
