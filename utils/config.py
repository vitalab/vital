from vital.utils.dataclasses import parameters


@parameters
class ResultTags:
    """ Class to gather the tags referring to the generic results stored in the HDF5 result files.

    Args:
        img: name of the tag referring to images.
        gt: name of the tag referring to groundtruths, used as reference when evaluating models' scores.
        pred: name of the tag referring to original predictions.
        post_pred: name of the tag referring to post processed predictions.
    """
    img: str = 'img'
    gt: str = 'gt'
    pred: str = 'pred'
    post_pred: str = 'post_pred'
    encoding: str = 'encoding'
