# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""AzureML ACFT Image finetune component - augmentation helper methods."""

import os
import yaml
from typing import List, Dict
import logging

# Need to update from absolute to relative import
from custom_augmentations import albumentations
from common_constants import (
    AugmentationConfigKeys,
    AugmentationConfigFileExts,
    AlbumentationParameterNames)


logger = logging.getLogger(__name__)


def save_augmentations_to_disk(output_folder: str, augmentation_dict: Dict) -> None:
    """ Save augmentation config to disk
    :param output_path: Path to save augmentation config
    :type output_path: str
    :param augmentation_dict: Augmentation config dictionary
    :type augmentation_dict: Dict

    :return: None
    """
    if augmentation_dict is None:
        logger.info("No augmentation config provided. Skipping saving augmentations to disk.")
    output_file = os.path.join(output_folder, AugmentationConfigKeys.OUTPUT_AUG_FILENAME)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_file, "w") as f:
        yaml.dump(augmentation_dict, f, default_flow_style=False)
        logger.info(f"Augmentations saved at {output_file}")


def load_augmentation_dict_from_yaml(config_path: str, skip_validation=False) -> dict:
    """
    This function expects a yaml file in following format, and load the augmentations config file into a dictionary

    augmentation_library_name: <albumentations>
    train:
        - <albumentations_function_name_1>:
              <function_1_parameter_1>: <value_1>
              <function_1_parameter_2>: <value_2>
              ...
        - <albumentations_function_name_2>:
              <function_2_parameter_1>: <value_3>
              <function_2_parameter_2>: <value_4>
              ...
        ...
    validation:
        - <albumentations_function_name_1>:
              <function_1_parameter_1>: <value_1>
              <function_1_parameter_2>: <value_2>
              ...
        - <albumentations_function_name_2>:
              <function_2_parameter_1>: <value_3>
              <function_2_parameter_2>: <value_4>
              ...
        ...

    :param config_path: Path to augmentation config
    :type config_path: str
    :param skip_validation: Whether to skip validation of augmentation config for mandatory keys
    :type skip_validation: bool
    :return: augmentation config dictionary containing
            function name, function params & values for
            train and validation
            {
                "augmentation_library_name": <"albumentations"> or <"kornia"> or <"torchvision">,
                "train": [{"function_name": {function_param_name: value, ...}, ...}],
                "validation": [{"function_name": {function_param_name: value, ...}, ...}],
            }
    :rtype: dict
    """
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as af:
        incoming_augmentation_dict = yaml.load(af, Loader=yaml.FullLoader)

    if skip_validation:
        return incoming_augmentation_dict

    incoming_keys = list(incoming_augmentation_dict.keys())
    if AugmentationConfigKeys.AUGMENTATION_LIBRARY_NAME not in incoming_keys:
        # Ensure augmentation_library_name is provided in config
        raise KeyError(
            f"{AugmentationConfigKeys.AUGMENTATION_LIBRARY_NAME} not in yaml keys: {incoming_keys}"
        )
    if AugmentationConfigKeys.TRAINING_PHASE_KEY not in incoming_keys:
        raise KeyError(
            f"{AugmentationConfigKeys.TRAINING_PHASE_KEY} not in yaml keys: {incoming_keys}"
        )

    if (
        len(incoming_keys) == 3
        and AugmentationConfigKeys.VALIDATION_PHASE_KEY not in incoming_keys
    ):
        # Check for validation key, only when there are >2 keys present
        raise KeyError(
            f"{AugmentationConfigKeys.VALIDATION_PHASE_KEY} not in yaml keys: {incoming_keys}"
        )

    return incoming_augmentation_dict


def load_augmentation_dict_from_config(config_path: str) -> dict:
    """Load the augmentations config file into a dictionary

    :param config_path: Path to augmentation config
    :typr config_path: str

    :return: augmentation config dictionary containing
            function name, function params & values for
            train and valid
            {
                "augmentation_library_name": <"albumentations"> or <"kornia"> or <"torchvision">,
                "train": {"function_name": {function_param_name: value, ...}, ...},
                "valid": {"function_name": {function_param_name: value, ...}, ...},
            }
    :rtype: dict
    """
    load_augmentation_dict_from_config_factory = {
        AugmentationConfigFileExts.YAML: load_augmentation_dict_from_yaml
    }
    config_file_type = os.path.splitext(config_path)[-1]
    if config_file_type not in load_augmentation_dict_from_config_factory:
        raise NotImplementedError(
            f"Augmentation config {config_path}, file type {config_file_type} is not supported."
        )
    return load_augmentation_dict_from_config_factory[config_file_type](
        config_path=config_path
    )


def get_transform_list(augmentation_list: List[Dict]) -> List[Dict]:
    """
    Given the list in transformation names, convert it into list of transformation objects.

    :param augmentation_list: List of transformations
    :type augmentation_list: List[Dict]

    :return List of transformation objects to be applied
    :rtype: List[Dict]

    Example:
    augmentation_list = [{
        "HorizontalFlip": {"p": 0.5}
    }]
    get_transform_list(augmentation_list) would return [albumentations.HorizontalFlip(p=0.5)]

    augmentation_list = [{
        "OneOf": {
            "p": 1,
            "transforms": {
                "HorizontalFlip": {"p": 0.5},
                "RandomResizedCrop": {"width": 224, "height": 224, "p": 0.5}
            }
        }
    }]
    get_transform_list(augmentation_list) would return [
        albumentations.OneOf(p=1, transforms=[
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomResizedCrop(width=224, height=224, p=0.5)
        ])
    ]
    """
    tranform_list = []
    for augmentation_dict_item in augmentation_list:
        augmentation_function_name = list(augmentation_dict_item.keys())[0]
        augmentation_function_params_dict = augmentation_dict_item[augmentation_function_name]
        if AlbumentationParameterNames.TRANSFORMS_KEY in augmentation_function_params_dict:
            # If transforms list is present inside the augmentation, then iterate over the augmentations
            # This is required for compositional albumentation augmentations such as OneOf, Sequential, SomeOf etc.
            augmentation_function_params_dict[AlbumentationParameterNames.TRANSFORMS_KEY] = \
                get_transform_list(
                    augmentation_function_params_dict.get(AlbumentationParameterNames.TRANSFORMS_KEY, [])
            )

        # Append to list of transforms
        tranform_list.append(
            getattr(albumentations, augmentation_function_name)(
                **augmentation_function_params_dict
            )
        )
    return tranform_list


def get_transform(
        phase_key: str,
        augmentation_dict: Dict,
        is_bbox_required: bool = False
) -> albumentations.core.composition.Compose:
    """ Get transform for specified phase <train/valid>

    :param phase_key: Name of the phase (one of train, valid) for which transform is required.
    :type phase_key: str
    :param augmentation_dict: Dictionary containing augmentation configuration
    :type augmentation_dict: Dict
    :param is_bbox_required: Flag to indicate if bbox is required for the transform
    :type is_bbox_required: bool

    :return: Albumentation transform
    :rtype: albumentations.core.composition.Compose
    """
    tranform_list = get_transform_list(augmentation_dict[phase_key])

    # Albumentation requires extra bounding box related parameters for processing bbox.
    extra_params = {
        AlbumentationParameterNames.BBOX_PARAMS: albumentations.BboxParams(
            format=AlbumentationParameterNames.PASCAL_VOC, label_fields=[AlbumentationParameterNames.CLASS_LABELS]
        )} if is_bbox_required else dict()
    albumentations_transforms = albumentations.Compose(transforms=tranform_list, **extra_params)
    return albumentations_transforms
