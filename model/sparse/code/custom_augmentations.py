# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Custom albumentation augmentations."""

import albumentations
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Union, Callable
from albumentations.pytorch import transforms
from albumentations import DualTransform
from common_constants import AlbumentationParameterNames

BoxInternalType = Tuple[float, float, float, float]


class RandomExpand(DualTransform):
    """ Expand the image, bbox and mask by a random ratio and fill the surrounding space with the mean of ImageNet.
        This is intended to detect smaller objects.
    """

    @property
    def targets_as_params(self):
        """ targets which are required for the transformation as parameters.
        These targets are passed to get_params_dependent_on_targets method as params.
        """
        return [AlbumentationParameterNames.IMAGE]

    def get_params_dependent_on_targets(self, params: Dict) -> Tuple:
        """ Calcuate the parameters for the transformation which are dependent on the input image.
        :params: Dict of input image
        :type params: Dict
        :return: Tuple of parameters
        :rtype: Tuple
        """
        img = params[AlbumentationParameterNames.IMAGE]
        height, width, _ = img.shape

        ratio = random.uniform(1, 2)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        top = random.randint(0, new_height - height)
        left = random.randint(0, new_width - width)
        return {
            AlbumentationParameterNames.ORIGINAL_WIDTH: width,
            AlbumentationParameterNames.ORIGINAL_HEIGHT: height,
            AlbumentationParameterNames.NEW_WIDTH: new_width,
            AlbumentationParameterNames.NEW_HEIGHT: new_height,
            AlbumentationParameterNames.NEW_LEFT: left,
            AlbumentationParameterNames.NEW_TOP: top
        }

    def apply(self,
              img: np.ndarray,
              original_width: int,
              original_height: int,
              new_width: int,
              new_height: int,
              new_left: int,
              new_top: int,
              **params) -> np.ndarray:
        """ Overrides the DualTransform.apply method. This would apply the
        transformation to the input image.
        :param image: image
        :type image: np.ndarray
        :param original_width: Original width of the image
        :type original_width: int
        :param original_height: Original height of the image
        :type original_height: int
        :param new_width: New width after expanding the image
        :type new_width: int
        :param new_height: New height after expanding the image
        :type new_height: int
        :param new_left: New left of the image after expansion
        :type new_left: int
        :param new_top: New top of the image after expansion
        :type new_top: int
        :return: transformed image
        :rtype: np.ndarray
        """
        # Todo: Replace imagenet mean read from model's config file
        imagenet_mean = [0.485, 0.456, 0.406]

        tensor_image = transforms.img_to_tensor(img)

        # place an image in a larger mean image
        new_image = torch.ones((3, new_height, new_width), dtype=torch.float)
        new_image[:, :, :] *= torch.FloatTensor(imagenet_mean).unsqueeze(1).unsqueeze(2)
        new_image[:, new_top: new_top + original_height, new_left:new_left + original_width] = tensor_image
        # convert image(C, H, W) to (H, W, C)
        return new_image.numpy().transpose(1, 2, 0)

    def apply_to_bbox(self,
                      bbox: BoxInternalType,
                      original_width: int,
                      original_height: int,
                      new_width: int,
                      new_height: int,
                      new_left: int,
                      new_top: int,
                      **params) -> BoxInternalType:
        """ Overrides the DualTransform.apply_to_bbox method. This would apply
        the transformations to each bounding box separately.
        :param image: bbox
        :type image: Tuple[float, float, float, float]
        :param original_width: Original width of the image
        :type original_width: int
        :param original_height: Original height of the image
        :type original_height: int
        :param new_width: New width after expanding the image
        :type new_width: int
        :param new_height: New height after expanding the image
        :type new_height: int
        :param new_left: New left of the image after expansion
        :type new_left: int
        :param new_top: New top of the image after expansion
        :type new_top: int
        :return: transformed bounding box
        :rtype: Tuple[float, float, float, float]
        """
        # Note: Albumentation normalizes the input bboxes to keep the scale in range[0,1].
        denormalized_bbox = (
            bbox[0] * original_width, bbox[1] * original_height,
            bbox[2] * original_width, bbox[3] * original_height
        )
        new_bbox = torch.Tensor(denormalized_bbox)
        new_bbox += torch.FloatTensor([new_left, new_top, new_left, new_top])
        new_bbox = tuple(new_bbox)
        normalized_box = (
            new_bbox[0] / new_width, new_bbox[1] / new_height,
            new_bbox[2] / new_width, new_bbox[3] / new_height
        )
        return normalized_box

    def apply_to_masks(self, masks: List[np.ndarray],
                       original_width: int,
                       original_height: int,
                       new_width: int,
                       new_height: int,
                       new_left: int,
                       new_top: int,
                       **params) -> List[np.ndarray]:
        """ Overrides the DualTransform.apply_to_masks method. This would apply
        the transformations to the masks.
        :param masks: list of masks
        :type image: List[np.ndarray]
        :param original_width: Original width of the image
        :type original_width: int
        :param original_height: Original height of the image
        :type original_height: int
        :param new_width: New width after expanding the image
        :type new_width: int
        :param new_height: New height after expanding the image
        :type new_height: int
        :param new_left: New left of the image after expansion
        :type new_left: int
        :param new_top: New top of the image after expansion
        :type new_top: int
        :return: transformed masks
        :rtype: List[np.ndarray]
        """
        # if there are masks, align them with the image
        new_masks = None
        if masks is not None:
            new_masks = np.zeros((len(masks), new_height, new_width), dtype=np.float32)
            new_masks[:, new_top:new_top + original_height, new_left:new_left + original_width] = masks
        return new_masks

    def get_transform_init_args_names(self) -> Tuple:
        """ Returns a tuple of arguments which are expected in the init method of the
        transformations.
        """
        return ()


class ConstraintResize(DualTransform):
    """
    Given the scale<min_size, max_size>, the image will be rescaled as large as possible within the scale.
    The image size will be constraint so that the max edge is no longer than max_size and
    short edge is no longer than min_size.
    """
    def __init__(self, img_scale: Union[List[int], List[Tuple]],
                 keep_ratio: bool = True, always_apply: bool = False, p: float = 0.5):
        """
        :param img_scale: Image scale for resizing. [min_size, max_size] or multiple (min_size, max_size) to randomly
        select from.
        :type img_scale: (List[int] or list[tuple(int, int)]).
        :param keep_ratio: Whether to keep the aspect ratio.
        :type keep_ratio: Boolean
        :param always_apply: Whether to apply the transformation always irrespective of the parameter 'p'.
        :type always_apply: Boolean
        :param p: Probability to apply the transform.
        :type p: float
        """
        super(ConstraintResize, self).__init__(always_apply, p)
        self.img_scale = img_scale
        self.min_size = min(img_scale)
        self.max_size = max(img_scale)
        self.keep_ratio = keep_ratio

    def _random_select(self):
        """Randomly select an img_scale from given candidates.
        Args:
            img_scales (list[tuple]): Images scales for selection.
        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(self.img_scale))
        selected_img_scale = self.img_scale[scale_idx]
        return selected_img_scale, scale_idx

    def _get_new_size(self, img: np.ndarray) -> Tuple[int, int, int, int, float, float]:
        """
        Calcuate the final size of the output image
        :param img: Input image
        :type img: np.ndarray
        :return a tuple of (current_height, current width,
                            new_height, new_width,
                            width_scale_factor, height_scale_factor)
        """

        h, w = img.shape[:2]
        if isinstance(self.img_scale[0], list) or isinstance(self.img_scale[0], tuple):
            randomly_selected_scale, _ = self._random_select()
            self.min_size = min(randomly_selected_scale)
            self.max_size = max(randomly_selected_scale)

        if self.keep_ratio:
            img_min_size = min(img.shape[:2])
            img_max_size = max(img.shape[:2])
            # Todo: Add support for min_size and max_size as a list also
            scale_factor = min(self.min_size / img_min_size, self.max_size / img_max_size)
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
        else:
            new_h = self.max_size if h > w else self.min_size
            new_w = self.min_size if h > w else self.max_size
        w_scale = new_w / w
        h_scale = new_h / h
        return w, h, new_w, new_h, w_scale, h_scale

    @property
    def targets_as_params(self):
        """ targets which are required for the transformation as parameters.
        Overrides the DualTransform.targets_as_params property.
        These targets are passed to get_params_dependent_on_targets method as params.
        """
        return [AlbumentationParameterNames.IMAGE]

    @property
    def targets(self) -> Dict[str, Callable]:
        """ targets for the augmentation such as image, mask, bbox, keypoints etc.
        Overrides the DualTransform.targets property.
        """
        super_targets = super().targets
        return {
            **super_targets,
            AlbumentationParameterNames.IMAGE_METADATA: self.apply_to_metadata
        }

    def get_params_dependent_on_targets(self, params: Dict) -> Tuple:
        """ Calcuate the parameters for the transformation which are dependent on the input image.
        Overrides the DualTransform.get_params_dependent_on_targets method. This would apply
        :params: Dict of input image
        :type params: Dict
        :return: Tuple of parameters
        :rtype: Tuple
        """
        img = params[AlbumentationParameterNames.IMAGE]
        width, height, new_w, new_h, w_scale, h_scale = self._get_new_size(img)
        return {
            AlbumentationParameterNames.ORIGINAL_WIDTH: width,
            AlbumentationParameterNames.ORIGINAL_HEIGHT: height,
            AlbumentationParameterNames.NEW_WIDTH: new_w,
            AlbumentationParameterNames.NEW_HEIGHT: new_h,
            AlbumentationParameterNames.WIDTH_SCALE: w_scale,
            AlbumentationParameterNames.HEIGHT_SCALE: h_scale
        }

    def apply(self,
              img: np.ndarray,
              new_width: int,
              new_height: int,
              **params) -> np.ndarray:
        """ Overrides the DualTransform.apply method. This would apply the
        transformation to the input image.
        :param image: image
        :type image: np.ndarray
        :param new_w: New width after resizing the image
        :param new_w: int
        :param new_h: New height after resizing the image
        :param new_h: int
        :return: transformed image
        :rtype: np.ndarray
        """

        return albumentations.resize(img, new_height, new_width)

    def apply_to_metadata(self,
                          metadata: Dict,
                          new_width: int,
                          new_height: int,
                          **params) -> Dict:
        """ Capture the resized image shape in the metadata
        :param metadata: Image metadata
        :type metadata: Dict
        :param new_w: New width after resizing the image
        :param new_w: int
        :param new_h: New height after resizing the image
        :param new_h: int
        :return: transformed metadata
        :rtype: Dict
        """
        resize_shape = {
            AlbumentationParameterNames.RESIZED_WIDTH: new_width,
            AlbumentationParameterNames.RESIZED_HEIGHT: new_height
        }
        return {**metadata, **resize_shape}

    def apply_to_bbox(self, bbox: BoxInternalType,
                      original_width: int,
                      original_height: int,
                      new_width: int,
                      new_height: int,
                      width_scale: float,
                      height_scale: float,
                      **params) -> BoxInternalType:
        """ Overrides the DualTransform.apply_to_bbox method. This would apply
        the transformations to each bounding box separately.
        :param image: bbox
        :type image: Tuple[float, float, float, float]
        :return: transformed bounding box
        :rtype: Tuple[float, float, float, float]
        """

        denormalized_bbox = (
            bbox[0] * original_width, bbox[1] * original_height,
            bbox[2] * original_width, bbox[3] * original_height
        )
        new_bbox = self.resize_bbox(denormalized_bbox, width_scale, height_scale)
        new_bbox = tuple(new_bbox)
        normalized_box = (
            new_bbox[0] / new_width, new_bbox[1] / new_height,
            new_bbox[2] / new_width, new_bbox[3] / new_height
        )
        return normalized_box

    def resize_bbox(cls, bbox: BoxInternalType,
                    w_scale: float, h_scale: float) -> Tuple[float, float, float, float]:
        """ Resize the bounding box
        :param bbox: bbox
        :type bbox: Tuple[float, float, float, float]
        :param w_scale: width scale factor
        :type w_scale: float
        :param h_scale: height scale factor
        :type h_scale: float
        :return: transformed bounding box
        :rtype: Tuple[float, float, float, float]
        """
        scale_multiplier = np.array([w_scale, h_scale, w_scale, h_scale])
        np_bbox = np.array(bbox)
        return np_bbox * scale_multiplier

    def apply_to_masks(self, masks: List[np.ndarray], new_width: int,
                       new_height: int, **params) -> List[np.ndarray]:
        """ Overrides the DualTransform.apply_to_masks method. This would apply
        the transformations to the masks.
        :param masks: list of masks
        :type image: List[np.ndarray]
        :return: transformed masks
        :rtype: List[np.ndarray]
        """

        if masks is None:
            return masks
        rescaled_masks = []
        for mask in masks:
            rescaled_masks.append(albumentations.resize(mask, new_height, new_width))
        return rescaled_masks

    def get_transform_init_args_names(self) -> Tuple:
        """ Returns a tuple of arguments which are expected in the init method of the
        transformations.
        """

        return (
            "img_scale",
            "keep_ratio"
        )


albumentations.RandomExpand = RandomExpand
albumentations.ConstraintResize = ConstraintResize
