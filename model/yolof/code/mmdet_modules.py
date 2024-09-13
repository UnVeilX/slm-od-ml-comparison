# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""MMDetection modules."""

import logging
import torch
import numpy as np

from dataclasses import dataclass
from mmengine.config import Config
from mmengine.utils import concat_list
from mmengine.structures import InstanceData
try:
    # mmdet = 3.x
    from mmdet.structures import DetDataSample
except ImportError:
    # mmdet = 2.x
    DetDataSample = None
from pathlib import Path
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Union, Any, Tuple
from common_constants import MmDetectionDatasetLiterals, MMdetModes, MmDetectionConfigLiterals
from common_utils import get_current_device

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Dataclass for maintaining the metadata dictionary as required for MM detection models.
    The keys of metadata dictionary is same as the property name."""

    ori_shape: Tuple[int, int, int]         # Dimension of the transformed image H x W x C. This is the shape of the
    # image after resizing and padding. If we set it to the original image shape i.e. shape before transformation
    # then the MMD model is not able to learn and the eval_mAP is always 0.0.
    img_shape: Tuple[int, int, int] = None  # Dimension of the image after ConstraintResize transformation.
    pad_shape: Tuple[int, int, int] = None  # Dimension of the tranformed image after padding. Please note that padding
    # is applied after ConstraintResize transformation, so pad_shape is always greater than or equal to img_shape.
    raw_dimensions: Tuple[int, int] = None   # Dimension of the raw image H x W. This is used to resize the predicted
    # mask to the original image size.
    scale_factor: np.ndarray = np.array([1, 1])
    flip: bool = False
    flip_direction: str = None
    filename: str = None
    ori_filename: str = None
    border: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self):
        """If image shape after resizing and padding is not provided then assign it with original shape"""
        self.img_shape = self.img_shape or self.ori_shape
        self.pad_shape = self.pad_shape or self.ori_shape


class ObjectDetectionModelWrapper(nn.Module):
    """Wrapper class over object detection model of MMDetection."""
    def __init__(
        self,
        mm_object_detection_model: nn.Module,
        config: Config,
        model_name_or_path: str = None,
    ):
        """Wrapper class over object detection model of MMDetection.

        :param mm_object_detection_model: MM object detection model
        :type mm_object_detection_model: nn.Module
        :param config: MM Detection model configuration
        :type config: MMCV Config
        :param model_name_or_path: model name or path
        :type model_name_or_path: str
        """

        super().__init__()
        self.model = mm_object_detection_model
        self.config = config
        self.model_name = Path(model_name_or_path).stem
        self.lang_model = hasattr(self.config, MmDetectionConfigLiterals.LANG_MODEL_NAME)
        self.text = None
        if self.lang_model:
            try:
                classes = self.config.test_dataloader.dataset.metainfo.CLASSES
                self.text = ". ".join(classes)
            except Exception as e:
                logger.error("Unable to fetch classes information from config file")
                raise e

    @classmethod
    def _pad_sequence(cls, sequences: Tensor, padding_value: float = -1, batch_first: bool = True) -> Tensor:
        """
        It stacks a list of Tensors sequences, and pads them to equal length.
        :param sequences: list of variable length sequences.
        :type sequences: Tensor
        :param padding_value: value for padded elements
        :type padding_value: float
        :param batch_first: output will be in B x T x * if True, or in T x B x * otherwise
        :type batch_first: bool
        :return: Tensor of size ``B x T x *`` if batch_first is True
        :rtype: Tensor
        """
        rt_tensor = pad_sequence(sequences, padding_value=padding_value, batch_first=batch_first)
        rt_tensor = rt_tensor.to(device=get_current_device())
        return rt_tensor

    def _get_bboxes_and_labels(self, predictions):
        """
        Get bounding boxes and labels from the predictions.
        :param predictions: predictions from the model
        :type predictions: mmdet.structures.DetDataSample
        :return: bounding boxes and labels
        :rtype: Tensor, Tensor
        """
        predictions = predictions.detach()
        pred = predictions.pred_instances
        height, width = predictions.img_shape

        scores = pred.scores.unsqueeze(dim=1)
        img_box = torch.tensor([width, height, width, height], device=scores.device)
        bboxes = pred.bboxes / img_box
        bboxes = torch.concat((bboxes, scores), dim=1)

        return bboxes, pred.labels

    def _organize_predictions_for_trainer(
        self, batch_predictions: List[DetDataSample]
    ) -> Dict[str, Tensor]:
        """
        Transform the batch of predicted labels as required by the HF trainer.
        :param batch_predictions: batch of predicted labels
        :type batch_predictions: List of bbox list for each image
        :return: Dict of predicted labels in tensor format
        :rtype: Dict[str, Tensor]

        Note: Same reasoning like _organize_ground_truth_for_trainer function but for predicted label
        """
        batch_bboxes, batch_labels = [], []
        for predictions in batch_predictions:
            bboxes, labels = self._get_bboxes_and_labels(predictions)
            batch_bboxes.append(bboxes)
            batch_labels.append(labels)

        output = dict()
        output[MmDetectionDatasetLiterals.BBOXES] = ObjectDetectionModelWrapper._pad_sequence(batch_bboxes)
        output[MmDetectionDatasetLiterals.LABELS] = ObjectDetectionModelWrapper._pad_sequence(batch_labels)
        return output

    @classmethod
    def prepare_inputs_for_model_forward(cls, data, is_train=False, text=None):
        """ Prepare inputs as required for mmdetection.

        :param data: list of dictionaries from od collate func
        :type data: list of Dict
        :param is_train: whether the data is for training or not
        :type is_train: bool
        :param text: str representing all the classes of the data joined by ". ".
        :type text: str
        :return: image tensor and list of datasamples
        :rtype: Tensor, List[mmdet.structures.DetDataSample]
        """

        inputs = data.get(MmDetectionDatasetLiterals.IMG)
        batch_data_samples = []
        for i in range(len(inputs)):
            img_metainfo = data[MmDetectionDatasetLiterals.IMG_METAS][i]
            img_metainfo[MmDetectionDatasetLiterals.IMAGE_SHAPE] = \
                img_metainfo[MmDetectionDatasetLiterals.IMAGE_SHAPE][:2]
            img_metainfo.update({"batch_input_shape": inputs.shape[2:]})
            if DetDataSample is None:
                raise SystemError("DetDataSample is set to be None. "
                                  "For detection tasks, please check installation of mmdet 3.x installation. "
                                  "This function should not be called n video-multi-object-tracking tasks.")
            data_sample = DetDataSample(metainfo=img_metainfo)

            if is_train:
                gt_instances = InstanceData(metainfo=img_metainfo)
                if MmDetectionDatasetLiterals.GT_BBOXES in data:
                    gt_instances.bboxes = data[MmDetectionDatasetLiterals.GT_BBOXES][i]
                    gt_instances.labels = data[MmDetectionDatasetLiterals.GT_LABELS][i]
                # gt_masks would be present for instance segmentation.
                if MmDetectionDatasetLiterals.GT_MASKS in data:
                    gt_instances.masks = data[MmDetectionDatasetLiterals.GT_MASKS][i]
                data_sample.gt_instances = gt_instances
            if text:
                data_sample.text = text
                data_sample.custom_entities = True
            batch_data_samples.append(data_sample)

        return inputs, batch_data_samples

    def forward(
        self, **data
    ) -> Union[Dict[str, Any], Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Model forward pass for training and validation mode
        :param data: Input data to model
        :type data: Dict
        :return: A dictionary of loss components in training mode OR Tuple of dictionary of predicted and ground
        labels in validation mode
        :rtype: Dict[str, Any] in training mode; Tuple[Tensor, Dict[str, Tensor]] in validation mode;

        Note: Input data dictionary consist of
            img: Tensor of shape (N, C, H, W) encoding input images.
            img_metas: list of image info dict where each dict has: "img_shape", "scale_factor", "flip",
             and may also contain "filename", "ori_shape", "pad_shape", and "img_norm_cfg". For details on the values
             of these keys see `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes - Ground truth bboxes for each image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels - List of class indices corresponding to each box
            gt_crowds - List of "is crowds" (boolean) to each box
            gt_masks - List of masks (type BitmapMasks) for each image if task is instance_segmentation
        """
        # test mode
        inputs, batch_data_samples = ObjectDetectionModelWrapper.prepare_inputs_for_model_forward(data, text=self.text)
        batch_predictions = self.model(inputs, batch_data_samples, mode=MMdetModes.PREDICT)

        output: dict = self._organize_predictions_for_trainer(
            batch_predictions
        )

        return torch.asarray([], device=get_current_device()), output


class InstanceSegmentationModelWrapper(ObjectDetectionModelWrapper):
    """Wrapper class over mm instance segmentation model of MMDetection framework."""
    def __init__(
        self,
        mm_instance_segmentation_model: nn.Module,
        config: Config,
        model_name_or_path: str,
    ):
        """Wrapper class over mm instance segmentation model of MMDetection framework.

        :param mm_instance_segmentation_model: MM instance segmentation model
        :type mm_instance_segmentation_model: nn.Module
        :param config: MM Instance segmentation model configuration
        :type config: MMCV Config
        :param model_name_or_path: model name or path
        :type model_name_or_path: str
        """
        self.max_image_size = 0
        super(InstanceSegmentationModelWrapper, self).__init__(
            mm_instance_segmentation_model, config, model_name_or_path
        )

    def _organize_predictions_for_trainer(
        self, batch_predictions: List[DetDataSample]
    ) -> Dict[str, Tensor]:
        """
        Transform the batch of predicted labels as required by the HF trainer.
        :param batch_predictions: batch of predicted labels
        :type batch_predictions: List of bbox list for each image
        :return: Dict of predicted labels in tensor format
        :rtype: Dict[str, Tensor]

        Note: Same reasoning like _organize_ground_truth_for_trainer function but for predicted label
        """
        batch_bboxes, batch_labels, batch_masks = [], [], []
        batch_original_mask_shapes, batch_original_img_shapes = [], []
        for predictions in batch_predictions:
            bboxes, labels = self._get_bboxes_and_labels(predictions)
            pred_instances = predictions.pred_instances

            # HF Trainer stack the predictions of all batches together. Since prediction masks could be of
            # different size, We are padding the masks to the max possible image size and we are removing the
            # padding when we parse the instance segmentation outputs.
            masks = pred_instances.masks
            padded_masks = torch.empty(len(masks), self.max_image_size, self.max_image_size, dtype=torch.bool)
            padded_masks[:, :masks.shape[-2], :masks.shape[-1]] = masks
            original_mask_shape = masks.shape
            batch_masks.append(padded_masks)

            batch_bboxes.append(bboxes)
            batch_labels.append(labels)
            batch_original_mask_shapes.append(original_mask_shape)
            batch_original_img_shapes.append(predictions.raw_dimensions)

        output = dict()
        output[MmDetectionDatasetLiterals.BBOXES] = super()._pad_sequence(batch_bboxes)
        output[MmDetectionDatasetLiterals.LABELS] = super()._pad_sequence(batch_labels)
        output[MmDetectionDatasetLiterals.MASKS] = super()._pad_sequence(batch_masks)
        output[MmDetectionDatasetLiterals.RAW_DIMENSIONS] = \
            torch.tensor(batch_original_img_shapes, device=get_current_device())
        output[MmDetectionDatasetLiterals.RAW_MASK_DIMENSIONS] = \
            torch.tensor(batch_original_mask_shapes, device=get_current_device())
        return output
