from pathlib import Path
from typing import Any, Dict, Sequence, Union

import numpy as np
from tqdm.auto import tqdm

from vital.data.cardinal.config import CardinalTag
from vital.data.cardinal.utils.itertools import Views
from vital.utils.image.process import PostProcessor


def postprocess_masks(
    mask: np.ndarray, post_tag: str, postprocessing_ops: Sequence[PostProcessor]
) -> Union[np.ndarray, Dict[str, Any]]:
    """Post-process a batch of 2D segmentation masks.

    Args:
        mask: Batch of 2D segmentation masks.
        post_tag: Tag identifying the post-processed segmentation masks, in case the return value is a dictionary.
        postprocessing_ops: Post-processing operations to apply to the segmentation masks.

    Returns:
        Either the post-processed segmentation masks, or a dictionary containing the post-processed segmentation masks
        along with auxiliary results of the post-processing.
    """
    out = {post_tag: mask}
    for postprocessing_op in postprocessing_ops:
        post = postprocessing_op(out[post_tag])
        if isinstance(post, dict):
            out[post_tag] = post.pop(postprocessing_op.post_tag)
            out.update(post)
        elif isinstance(post, np.ndarray):
            out[post_tag] = post
        else:
            raise RuntimeError(
                f"Unsupported output type from postprocessor '{type(postprocessing_op)}' in the postprocessing loop. "
                f"Modify the postprocessor to return either a '{np.ndarray}' or a '{dict}', or remove the "
                f"postprocessor altogether."
            )
    if len(out) == 1:
        out = out.pop(post_tag)
    return out


def postprocess_views(
    views: Views,
    mask_tag: str,
    postprocessing_ops: Sequence[PostProcessor],
    output_root: Path,
    progress_bar: bool = False,
) -> None:
    """Post-process views' segmentation masks and saves the output to disk.

    Args:
        views: Iterator of views to post-process.
        mask_tag: Tag identifying the specific segmentation mask within the views to post-process.
        postprocessing_ops: Post-processing operations to apply to the views.
        output_root: Root of where to save the result of the post-processing.
        progress_bar: If ``True``, enables progress bars detailing the progress of the post-processing of each view in
            the collection.
    """
    post_tag = f"{CardinalTag.post}_{mask_tag}"

    views = views.values()
    if progress_bar:
        views = tqdm(views, desc=f"Post-processing views saving results to '{output_root}'", unit="view")
    for view in views:
        post = postprocess_masks(view.data[mask_tag], post_tag, postprocessing_ops)

        # Unpack the output of the post-processing
        if isinstance(post, dict):
            mask = post.pop(post_tag)
        else:
            mask = post
            post = {}

        # Add the output of the post-processing to the view object
        view.add_image(post_tag, mask, reference_img_tag=mask_tag)
        view.attrs[post_tag].update(post)

        view.save(output_root, include_tags=[post_tag])
