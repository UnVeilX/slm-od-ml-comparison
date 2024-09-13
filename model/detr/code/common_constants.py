# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Mlflow PythonModel wrapper helper constants."""

from dataclasses import dataclass
from mlflow.types import DataType


@dataclass
class AugmentationConfigKeys:
    """Keys in augmentation configs"""

    TRAINING_PHASE_KEY = "train"
    VALIDATION_PHASE_KEY = "validation"
    AUGMENTATION_LIBRARY_NAME = "augmentation_library_name"
    OUTPUT_AUG_FILENAME = "augmentations.yaml"
    ALBUMENTATIONS = "albumentations"


@dataclass
class AlbumentationParameterNames:
    """keys for Albumentations parameters"""

    TRANSFORMS_KEY = "transforms"
    BBOX_PARAMS = "bbox_params"
    PASCAL_VOC = "pascal_voc"
    CLASS_LABELS = "class_labels"
    IMAGE = "image"
    BBOX = "bbox"
    MASK = "mask"
    IMAGE_METADATA = "image_metadata"
    ORIGINAL_WIDTH = "original_width"
    ORIGINAL_HEIGHT = "original_height"
    NEW_WIDTH = "new_width"
    NEW_HEIGHT = "new_height"
    NEW_LEFT = "new_left"
    NEW_TOP = "new_top"
    WIDTH_SCALE = "width_scale"
    HEIGHT_SCALE = "height_scale"
    RESIZED_WIDTH = "resized_width"
    RESIZED_HEIGHT = "resized_height"


@dataclass
class AugmentationConfigFileExts:
    """Various Augmentation Config file types supported"""

    YAML = ".yaml"


class Tasks:
    "Tasks supported for All Frameworks"

    HF_MULTI_CLASS_IMAGE_CLASSIFICATION = "image-classification"
    HF_MULTI_LABEL_IMAGE_CLASSIFICATION = "image-classification-multilabel"
    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"
    MM_MULTI_OBJECT_TRACKING = "video-multi-object-tracking"
    HF_SD_TEXT_TO_IMAGE = "stable-diffusion-text-to-image"


class HFMiscellaneousLiterals:
    """HF miscellaneous constants"""

    PIXEL_VALUES = "pixel_values"
    DEFAULT_IMAGE_KEY = "image"
    IMAGE_FOLDER = "imagefolder"
    VAL = "val"
    ID2LABEL = "id2label"
    LABEL2ID = "label2id"


class HFConstants:
    """HF constants"""

    DEFAULT_DATALOADER_NUM_WORKERS = 6


class MLFlowSchemaLiterals:
    """MLFlow model signature related schema"""

    INPUT_IMAGE_KEY = "image_base64"
    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.binary
    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_VIDEO_DATA_TYPE = DataType.string
    INPUT_COLUMN_VIDEO = "video"
    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_FILENAME = "filename"
    OUTPUT_COLUMN_PROBS = "probs"
    OUTPUT_COLUMN_LABELS = "labels"
    OUTPUT_COLUMN_BOXES = "boxes"

    BATCH_SIZE_KEY = "batch_size"
    SCHEMA_SIGNATURE = "signature"
    TRAIN_LABEL_LIST = "train_label_list"
    WRAPPER = "images_model_wrapper"
    THRESHOLD = "threshold"

    INPUT_COLUMN_PROMPT_DATA_TYPE = DataType.string
    INPUT_COLUMN_PROMPT = "prompt"
    OUTPUT_COLUMN_IMAGE_TYPE = DataType.binary
    OUTPUT_COLUMN_IMAGE = "generated_image"
    OUTPUT_COLUMN_NSFW_FLAG_TYPE = DataType.boolean
    OUTPUT_COLUMN_NSFW_FLAG = "nsfw_content_detected"


class MMDetLiterals:
    """MMDetection constants"""

    CONFIG_PATH = "config_path"
    WEIGHTS_PATH = "weights_path"
    AUGMENTATIONS_PATH = "augmentations_path"
    METAFILE_PATH = "model_metadata"
    MODEL_DEFAULTS_PATH = "model_defaults_path"


class MMdetModes:
    """MMDetection forward mode constants"""

    LOSS = "loss"
    PREDICT = "predict"


class MmDetectionDatasetLiterals:
    """MMDetection dataset constants"""

    IMG = "img"
    IMG_METAS = "img_metas"
    GT_BBOXES = "gt_bboxes"
    GT_LABELS = "gt_labels"
    GT_CROWDS = "gt_crowds"
    GT_MASKS = "gt_masks"
    MASKS = "masks"
    BBOXES = "bboxes"
    LABELS = "labels"
    IMAGE_SHAPE = "img_shape"
    IMAGE_ORIGINAL_SHAPE = "ori_shape"
    PAD_SHAPE = "pad_shape"
    RAW_DIMENSIONS = "raw_dimensions"
    RAW_MASK_DIMENSIONS = "raw_mask_dimensions"
    SCALE_FACTOR = "scale_factor"
    DUMMY_LABELS = "dummy_labels"
    BATCH_DATA_SAMPLES = "batch_data_samples"
    ORIGINAL_GT_BBOXES = "original_gt_bboxes"
    ORIGINAL_GT_MASKS = "original_gt_masks"


class MmTrackingDatasetLiterals(MmDetectionDatasetLiterals):
    """MMTracking dataset constants"""

    DET_BBOXES = "det_bboxes"
    DET_LABELS = "det_labels"
    TRACK_BBOXES = "track_bboxes"
    TRACK_LABELS = "track_labels"
    INSTANCE_ID = "instance_id"

    IMG_INFO = "img_info"
    FRAME_ID = "frame_id"
    VIDEO_URL = "video_url"


class ODLiterals:
    """OD constants"""

    LABEL = "label"
    BOXES = "boxes"
    SCORE = "score"
    BOX = "box"
    TOP_X = "topX"
    TOP_Y = "topY"
    BOTTOM_X = "bottomX"
    BOTTOM_Y = "bottomY"
    POLYGON = "polygon"


class MmDetectionConfigLiterals:
    """MMDetection config constants"""

    NUM_CLASSES = "num_classes"
    BOX_SCORE_THRESHOLD = "score_thr"
    LANG_MODEL_NAME = "lang_model_name"


class MetricsLiterals:
    """Azureml Metrics Literals"""

    SCORES = "scores"
    CLASSES = "classes"
    METRICS_COMPUTER = "metrics_computer"
    METRICS = "metrics"


class TrainingDefaultsConstants:
    """Training Defaults Constants"""

    MODEL_DEFAULTS_FILE = "model_defaults.json"


class TrainingLiterals:
    """Training related literals."""

    CHECKPOINT = "checkpoint"
    PRECISION = "precision"
    BATCH_OUTPUT_PATH = "AZUREML_BI_OUTPUT_PATH"
    FP16_INFERENCE = "FP16_INFERENCE"


class DistributedConstants:
    """Distributed Constants"""

    LOCAL_RANK = "LOCAL_RANK"


class DatatypeLiterals:
    """Literals related to data type."""

    IMAGE_FORMAT = "PNG"
    STR_ENCODING = "utf-8"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_artifacts"
    MODEL_NAME = "model_name"
    MODEL_FILE = "MLmodel"
    SAVED_MODEL_PATH = "saved_model_path"
    CHECKPOINT = "checkpoint"


class MLflowMetadataLiterals:
    """MLflow metadata related literals."""

    METADATA = "metadata"
    APPLY_LORA = "apply_lora"
    BASE_MODEL_NAME = "base_model_name"
    AZUREML_BASE_IMAGE = "azureml.base_image"


class SDLiterals:
    """Stable Diffusion specific settings literals."""

    TOKENIZER_NAME_OR_PATH = "tokenizer_name_or_path"
    REVISION = "revision"
    NON_EMA_REVISION = "non_ema_revision"
    UNET = "unet"
    VAE = "vae"
    OFFSET_NOISE = "offset_noise"
    TEXT_ENCODER_USE_ATTENTION_MASK = "text_encoder_use_attention_mask"
    WITH_PRIOR_PRESERVATION = "with_prior_preservation"
    SNR_GAMMA = "snr_gamma"
    PRE_COMPUTE_TEXT_EMBEDDINGS = "pre_compute_text_embeddings"
    PRIOR_LOSS_WEIGHT = "prior_loss_weight"
    CLASS_LABELS_CONDITIONING = "class_labels_conditioning"

    SCHEDULER = "scheduler"
    SCHEDULER_TYPE = "scheduler_type"
    SCHEDULER_PATH = "scheduler_path"
    SCHEDULER_CLASS_NAME = "_class_name"

    TOKENIZER = "tokenizer"
    TOKENIZER_SUBFOLDER = "tokenizer_subfolder"

    TEXT_ENCODER = "text_encoder"
    TEXT_ENCODER_TYPE = "text_encoder_type"
    TEXT_ENCODER_SUBFOLDER = "text_encoder_subfolder"
    CLIP_TEXT_MODEL = "CLIPTextModel"
    T5ENCODER_MODEL = "T5EncoderModel"

    NUM_IMAGES_PER_PROMPT = "num_images_per_prompt"
    NEGATIVE_PROMPT = "negative_prompt"
    HEIGHT = "height"
    WIDTH = "width"
    GUIDANCE_SCALE = "guidance_scale"
    NUM_INFERENCE_STEPS = "num_inference_steps"


class SDSettingParameters:
    """A class for default settings."""

    TIMESTEPS = "timesteps"
    DEFAULT_PRIOR_LOSS_WEIGHT = 1.0
    PRECISION = 32
    DEFAULT_SCHEDULER = "DDPMScheduler"
    SCHEDULER_CONFIG = "scheduler_config.json"
    ADAPTER_MODEL_FILE = "adapter_model.bin"
    ADAPTER_CONFIG_FILE = "adapter_config.json"

    NUM_IMAGES_PER_PROMPT = 1
    NEGATIVE_PROMPT = [""]
    HEIGHT = 512
    WIDTH = 512
    GUIDANCE_SCALE = 7.5
    NUM_INFERENCE_STEPS = 50

    NUM_IMAGES_PER_PROMPT_DTYPE = "long"
    NEGATIVE_PROMPT_DTYPE = "string"
    HEIGHT_DTYPE = "long"
    WIDTH_DTYPE = "long"
    GUIDANCE_SCALE_DTYPE = "double"
    NUM_INFERENCE_STEPS_DTYPE = "long"


class SDPredictionType:
    """Prediction type literals."""

    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"


class SDDataLiterals:
    """Data specific literals."""

    INPUT_IDS = "input_ids"
    PIXEL_VALUES = "pixel_values"
    ATTENTION_MASK = "attention_mask"
