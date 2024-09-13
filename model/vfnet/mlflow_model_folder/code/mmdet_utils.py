# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Mlflow PythonModel wrapper helper scripts."""

import logging
import numpy as np
import torch

from datasets import load_dataset
from dataclasses import asdict
from transformers import Trainer, TrainingArguments
from typing import Dict, List, Callable, Tuple

from common_constants import (HFMiscellaneousLiterals,
                              Tasks,
                              MmDetectionDatasetLiterals,
                              ODLiterals,
                              AlbumentationParameterNames)
from mmdet_modules import ImageMetadata
from custom_augmentations import ConstraintResize

logger = logging.getLogger(__name__)


def _parse_object_detection_output(output: Dict[str, np.ndarray], id2label: Dict[int, str]) -> List[Dict]:
    proc_op = []
    for bboxes, labels in zip(output[MmDetectionDatasetLiterals.BBOXES],
                              output[MmDetectionDatasetLiterals.LABELS]):
        curimage_preds = {ODLiterals.BOXES: []}
        for bbox, label in zip(bboxes, labels):
            if label >= 0:
                curimage_preds[ODLiterals.BOXES].append({
                    ODLiterals.BOX: {
                        ODLiterals.TOP_X: float(bbox[0]),
                        ODLiterals.TOP_Y: float(bbox[1]),
                        ODLiterals.BOTTOM_X: float(bbox[2]),
                        ODLiterals.BOTTOM_Y: float(bbox[3]),
                    },
                    ODLiterals.LABEL: id2label[label],
                    ODLiterals.SCORE: float(bbox[4]),
                })
        proc_op.append(curimage_preds)
    return proc_op


def _parse_instance_segmentation_output(output: Dict[str, np.ndarray], id2label: Dict[int, str]) -> List[Dict]:
    from masktools import convert_mask_to_polygon
    proc_op = []
    for bboxes, labels, masks, raw_image_dimension, raw_mask_dimension in zip(
        output[MmDetectionDatasetLiterals.BBOXES],
        output[MmDetectionDatasetLiterals.LABELS],
        output[MmDetectionDatasetLiterals.MASKS],
        output[MmDetectionDatasetLiterals.RAW_DIMENSIONS],
        output[MmDetectionDatasetLiterals.RAW_MASK_DIMENSIONS]
    ):
        curimage_preds = {ODLiterals.BOXES: []}
        _, h, w = raw_mask_dimension
        # Remove the padding added to the masks to make them of equal size.
        masks = masks[:, :h, :w]
        # Postprocess predictions to resize masks to match original image dimensions
        masks = resize_masks(masks=masks, raw_image_dimension=raw_image_dimension)
        for bbox, label, mask in zip(bboxes, labels, masks):
            if label >= 0:
                box = {
                    ODLiterals.BOX: {
                        ODLiterals.TOP_X: float(bbox[0]),
                        ODLiterals.TOP_Y: float(bbox[1]),
                        ODLiterals.BOTTOM_X: float(bbox[2]),
                        ODLiterals.BOTTOM_Y: float(bbox[3]),
                    },
                    ODLiterals.LABEL: id2label[label],
                    ODLiterals.SCORE: float(bbox[4]),
                    ODLiterals.POLYGON: convert_mask_to_polygon(mask)
                }
                if len(box[ODLiterals.POLYGON]) > 0:
                    curimage_preds[ODLiterals.BOXES].append(box)
        proc_op.append(curimage_preds)
    return proc_op


def mmdet_run_inference_batch(
    test_args: TrainingArguments,
    model: torch.nn.Module,
    id2label: Dict[int, str],
    image_path_list: List,
    task_type: Tasks,
    test_transforms: Callable,
) -> List:
    """This method performs inference on batch of input images.

    :param test_args: Training arguments path.
    :type test_args: transformers.TrainingArguments
    :param image_processor: Preprocessing configuration loader.
    :type image_processor: transformers.AutoImageProcessor
    :param model: Pytorch model weights.
    :type model: transformers.AutoModelForImageClassification
    :param image_path_list: list of image paths for inferencing.
    :type image_path_list: List
    :param task_type: Task type of the model.
    :type task_type: constants.Tasks
    :param test_transforms: Transformations to apply to the test dataset before
                            sending it to the model.
    :param test_transforms: Callable
    :return: list of dict.
    :rtype: list
    """

    def collate_fn(examples: List[Dict[str, Dict]]) -> Dict[str, Dict]:
        # Filter out invalid examples
        valid_examples = [example for example in examples if example is not None]
        if len(valid_examples) != len(examples):
            if len(valid_examples) == 0:
                raise Exception("All images in the current batch are invalid.")
            else:
                num_invalid_examples = len(examples) - len(valid_examples)
                logger.info(f"{num_invalid_examples} invalid images found.")
                logger.info("Replacing invalid images with randomly selected valid images from the current batch")
                new_example_indices = np.random.choice(np.arange(len(valid_examples)), num_invalid_examples)
                for ind in new_example_indices:
                    # Padding the batch with valid examples
                    valid_examples.append(valid_examples[ind])
        # Capture original image size in format (height, width)
        original_image_sizes = [example[HFMiscellaneousLiterals.DEFAULT_IMAGE_KEY].size[::-1] for example in examples]

        resized_image_shapes = []
        # Pre processing Image
        if test_transforms is not None:
            for example in valid_examples:
                transformed_inputs = test_transforms(
                    image=np.array(example[HFMiscellaneousLiterals.DEFAULT_IMAGE_KEY]),
                    image_metadata=dict()
                )
                example[HFMiscellaneousLiterals.DEFAULT_IMAGE_KEY] = transformed_inputs[
                    HFMiscellaneousLiterals.DEFAULT_IMAGE_KEY]
                resized_width = transformed_inputs[AlbumentationParameterNames.IMAGE_METADATA][
                    AlbumentationParameterNames.RESIZED_WIDTH
                ]
                resized_height = transformed_inputs[AlbumentationParameterNames.IMAGE_METADATA][
                    AlbumentationParameterNames.RESIZED_HEIGHT
                ]
                resized_image_shapes.append((resized_height, resized_width))

        def to_tensor_fn(img):
            return torch.from_numpy(img.transpose(2, 0, 1)).to(dtype=torch.float)

        tensor_images = [to_tensor_fn(example[HFMiscellaneousLiterals.DEFAULT_IMAGE_KEY])
                         for example in valid_examples]
        # Create a batch of images of equal size which is equal to [N, C, H_max, W_max] where
        # H_max is the maximum height of the image in the current batch.
        # W_max is the maximum width of the image in the current batch.
        batched_images = make_batch(tensor_images)

        img_metas = []
        for i, example in enumerate(valid_examples):
            image = example[HFMiscellaneousLiterals.DEFAULT_IMAGE_KEY]
            if test_transforms:
                height, width, no_ch = image.shape
            else:
                width, height = image.size
                no_ch = len(image.getbands())
            img_metas.append(
                asdict(ImageMetadata(ori_shape=(height, width, no_ch),
                                     img_shape=resized_image_shapes[i] if test_transforms else None,
                                     raw_dimensions=tuple(original_image_sizes[i]),
                                     filename=f"test_{i}.jpg"))
            )
        # input to mmdet model should contain image and image meta data
        output = {
            MmDetectionDatasetLiterals.IMG: batched_images,
            MmDetectionDatasetLiterals.IMG_METAS: img_metas
        }

        return output

    inference_dataset = load_dataset(
        HFMiscellaneousLiterals.IMAGE_FOLDER,
        data_files={HFMiscellaneousLiterals.VAL: image_path_list}
    )
    inference_dataset = inference_dataset[HFMiscellaneousLiterals.VAL]

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=test_args,
        data_collator=collate_fn,
    )
    results = trainer.predict(inference_dataset)
    output = results.predictions[1]
    if task_type == Tasks.MM_OBJECT_DETECTION:
        return _parse_object_detection_output(output, id2label)
    elif task_type == Tasks.MM_INSTANCE_SEGMENTATION:
        return _parse_instance_segmentation_output(output, id2label)


def make_batch(images: List[torch.Tensor]) -> torch.Tensor:
    """Make batch for object detection and instance segmentation
    Hf Trainer expects a torch tensor for input images. For OD and IS, the images in the current batch can have
    different dimensions(due to constraintResize and other transformations applied at train time), we need to pad
    the images to make them of equal size. To create a batch of images, we first find the maximum height and
    maximum width of the images in the current batch. We then pad the images to match the maximum height and
    maximum width. We then stack the images to create a batch of images.

    :param images: list of Images
    :type images: List[torch.Tensor]
    :return: batch of Images of size [channels, height_max, width_max] where height_max and width_max are the maximum
    height and width of the images in the current batch
    :rtype: torch.Tensor
    """
    batch_width, batch_height = 0, 0
    for img in images:
        img_height, img_width = img.shape[-2:]
        batch_width = max(img_width, batch_width)
        batch_height = max(img_height, batch_height)

    padded_images = []
    for img in images:
        img_height, img_width = img.shape[-2:]
        if img_height != batch_height or img_width != batch_width:
            # Only pad images when the image height or width is not equal to batch height or width respectively.
            padded_images.append(pad_img(img, batch_width, batch_height))
        else:
            # No need to pad if the image height and width is equal to batch height and width respectively.
            padded_images.append(img)
    pixel_values = torch.stack(padded_images)
    return pixel_values


def pad_img(img: torch.Tensor,
            new_width: int,
            new_height: int) -> torch.Tensor:
    """Pad image and to match the new width and height
    :param img: input image
    :type img: torch.Tensor
    :param new_width: new width
    :type new_width: int
    :param new_height: new height
    :param new_height: int
    :return: padded image
    :rtype: torch.Tensor
    """
    img_height, img_width = img.shape[-2:]
    padded_img = torch.zeros((3, new_height, new_width), dtype=img.dtype)
    padded_img[:, :img_height, :img_width] = img
    return padded_img


def resize_masks(masks: List[np.ndarray], raw_image_dimension: List) -> np.ndarray:
    """Postprocess predictions to resize the masks to match original image dimensions
    :masks: instance segmentation masks
    :type masks: List[np.ndarray]
    :img_shape: image shape after resizing
    :type img_shape: List
    :raw_image_dimension: Dimensions of the original image before preprocessing
    :type raw_image_dimension: List
    :return: resized masks
    :return: np.ndarray
    """

    resizer = ConstraintResize(img_scale=(-1, -1))
    h_original, w_original = raw_image_dimension
    resized_masks = None
    if masks is not None:
        if masks.dtype != np.uint8:
            masks.dtype = np.uint8
        resized_masks = resizer.apply_to_masks(masks, w_original, h_original)
    return resized_masks
