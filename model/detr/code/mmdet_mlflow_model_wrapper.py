# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Mlflow PythonModel wrapper class that loads the Mlflow model, preprocess inputs and performs inference."""

import logging
import subprocess
import sys
import tempfile
import math
import albumentations

import mlflow
import pandas as pd
import torch
from transformers import TrainingArguments

from augmentation_helper import (
    load_augmentation_dict_from_config,
    get_transform
)
from common_constants import (AugmentationConfigKeys,
                              HFMiscellaneousLiterals,
                              HFConstants, Tasks,
                              MMDetLiterals,
                              MLFlowSchemaLiterals)
from common_utils import process_image, create_temp_file, get_current_device

logger = logging.getLogger(__name__)


def get_max_image_size(transforms_list: albumentations.Compose) -> int:
    """ For MMD model, read the transforms list and return the max image size that the image
        could be resized to by the ConstraintResize transformation.
        This is calculated from the img_scale attribute of the ConstraintResize
        and the pad_factor attribute of the PadIfNeeded transforms.

        :param transforms_list: list of transforms
        :type transforms_list: list
        :return: max image size
        :rtype: int
    """
    max_size, max_pad_divisor = 0, 1
    for transform in transforms_list.transforms:
        if transform.__class__.__name__ == "ConstraintResize":
            max_size = max(transform.img_scale)
        elif transform.__class__.__name__ == "PadIfNeeded":
            max_pad_divisor = max(transform.pad_width_divisor, transform.pad_height_divisor)

    max_img_size = math.ceil(max_size / max_pad_divisor) * max_pad_divisor
    return max_img_size


class ImagesMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow model wrapper for AutoML for Images models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """This method is called when the python model wrapper is initialized.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self.test_args = None
        self.test_transforms = None
        self.mmdet_run_inference_batch = None
        self._config = None
        self._model = None
        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """This method is called when loading a Mlflow model with pyfunc.load_model().

        :param context: Mlflow context containing artifacts that the model can use for inference.
        :type context: mlflow.pyfunc.PythonModelContext
        """
        logger.info("Inside load_context()")

        if self._task_type in [Tasks.MM_OBJECT_DETECTION, Tasks.MM_INSTANCE_SEGMENTATION]:
            # Install mmcv and mmdet using mim, with pip installation is not working
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet==3.3.0"])

            # importing mmdet/mmcv after installing using mim
            from mmengine.config import Config
            from mmengine.runner import load_checkpoint
            from mmdet.apis import init_detector
            from mmdet_modules import ObjectDetectionModelWrapper, InstanceSegmentationModelWrapper
            from mmdet_utils import mmdet_run_inference_batch
            self.mmdet_run_inference_batch = mmdet_run_inference_batch

            try:
                current_device = get_current_device()
                model_config_path = context.artifacts[MMDetLiterals.CONFIG_PATH]
                model_weights_path = context.artifacts[MMDetLiterals.WEIGHTS_PATH]
                self._config = Config.fromfile(model_config_path)
                self._model = init_detector(self._config, device=current_device)
                if self._task_type == Tasks.MM_INSTANCE_SEGMENTATION:
                    self._model = InstanceSegmentationModelWrapper(self._model, self._config, model_weights_path)
                elif self._task_type == Tasks.MM_OBJECT_DETECTION:
                    self._model = ObjectDetectionModelWrapper(self._model, self._config, model_weights_path)
                load_checkpoint(self._model, model_weights_path, map_location=current_device)
                logger.info("Model loaded successfully")
            except Exception:
                logger.warning("Failed to load the the model.")
                raise

            aug_config_path = context.artifacts[MMDetLiterals.AUGMENTATIONS_PATH]
            aug_config_dict = load_augmentation_dict_from_config(aug_config_path)
            self.test_transforms = get_transform(AugmentationConfigKeys.VALIDATION_PHASE_KEY,
                                                 aug_config_dict,
                                                 # Bbox is not required at test time
                                                 is_bbox_required=False)
            max_image_size = get_max_image_size(self.test_transforms)
            self._model.max_image_size = max_image_size
        else:
            raise ValueError(f"invalid task type {self._task_type}."
                             f"Supported tasks: {Tasks.MM_OBJECT_DETECTION, Tasks.MM_INSTANCE_SEGMENTATION}")

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame
    ) -> pd.DataFrame:
        """This method performs inference on the input data.

        :param context: Mlflow context containing artifacts that the model can use for inference.
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images for prediction.
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 String format.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["probs", "labels"] for classification and
        ["boxes"] for object detection, instance segmentation
        """
        task = self._task_type

        ngpus = torch.cuda.device_count()
        batch_size = len(input_data)
        if ngpus > 1:
            batch_size = int(math.ceil(batch_size // ngpus))

        logger.info(f"evaluating with batch_size: {batch_size} and n_gpus: {ngpus}")
        # arguments for Trainer
        test_args = TrainingArguments(
            output_dir=".",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=batch_size,
            dataloader_num_workers=HFConstants.DEFAULT_DATALOADER_NUM_WORKERS,
            dataloader_drop_last=False,
            remove_unused_columns=False
        )

        # process the images in image column
        processed_images = input_data.loc[:, [MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE]]\
            .apply(axis=1, func=process_image)

        # To Do: change image height and width based on kwargs.

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            image_path_list = (
                processed_images.iloc[:, 0]
                .map(lambda row: create_temp_file(row, tmp_output_dir))
                .tolist()
            )

            result = self.mmdet_run_inference_batch(
                test_args,
                model=self._model,
                id2label=self._config[HFMiscellaneousLiterals.ID2LABEL],
                image_path_list=image_path_list,
                task_type=task,
                test_transforms=self.test_transforms,
            )

        return pd.DataFrame(result)
