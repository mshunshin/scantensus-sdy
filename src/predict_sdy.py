import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os
import yaml
from pathlib import Path

from SDY_File import SDY_File

import torch

import torchvision.io

from Scantensus.labels.json import get_keypoint_names_and_colors_from_json

from ScantensusPT.utils import load_and_fix_state_dict
from ScantensusPT.utils.heatmap_to_label import heatmap_to_label
from ScantensusPT.utils.heatmaps import gaussian_blur2d_norm
from ScantensusPT.image import image_logit_overlay_alpha_t

################
NUM_CUDA_DEVICES = torch.cuda.device_count()
CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES', None)
################

####################
DEBUG = False
RUN = "flow-004"
###################

#### Load Config File ####
with open(f'./runs/{RUN}.yaml', "r") as yaml_f:
    config = yaml.full_load(yaml_f)


#### Details ####
PROJECT = config['project']
EXPERIMENT = config['experiment']


#### Host ####
HOST = os.environ['UNITY_HOST']
HOST_CONFIG_DICT = config['hosts'][HOST]


#### Host Config ####
USE_CUDA = HOST_CONFIG_DICT['use_cuda']
DISTRIBUTED_BACKEND = HOST_CONFIG_DICT['distributed_backend']
DDP_PORT = HOST_CONFIG_DICT['ddp_port']

if USE_CUDA:
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

#### setup ####
MANUAL_SEED = config['setup']['manual_seed']
SINGLE_INFER_WORKERS = config['setup']['single_infer_workers']

if DEBUG:
    SINGLE_INFER_WORKERS = 0


#### data ####
DATA_CONFIG_NAME = config['infer_data_config_name']
DATA_CONFIG_DICT = config['infer_data_configs'][DATA_CONFIG_NAME]


#### data folders ####
DATA_DIR = Path(HOST_CONFIG_DICT['data_dir'])

PNG_CACHE_PROJECT_FOLDER = DATA_CONFIG_DICT['png_cache_project_folder']
PNG_CACHE_DIR = DATA_DIR / "png-cache" / PNG_CACHE_PROJECT_FOLDER


#### image ####
IMAGE_CROP_SIZE = DATA_CONFIG_DICT['input']['image_crop_size']
#IMAGE_OUT_SIZE = DATA_CONFIG_DICT['input']['image_out_size']
PRE_POST = DATA_CONFIG_DICT['input']['pre_post']


### heatmaps ###
DOT_SD = DATA_CONFIG_DICT['heatmaps']['dot_sd']
CURVE_SD = DATA_CONFIG_DICT['heatmaps']['curve_sd']

DOT_WEIGHT_SD = DATA_CONFIG_DICT['heatmaps']['dot_weight_sd']
CURVE_WEIGHT_SD = DATA_CONFIG_DICT['heatmaps']['curve_weight_sd']

DOT_WEIGHT = DATA_CONFIG_DICT['heatmaps']['dot_weight']
CURVE_WEIGHT = DATA_CONFIG_DICT['heatmaps']['curve_weight']

SUBPIXEL = DATA_CONFIG_DICT['heatmaps']['subpixel']

#### model ####
MODEL = config['model']['model_name']
MODEL_CLASS = config['model']['model_class']
MODEL_CONFIG_NAME = config['model']['model_config_name']
MODEL_CONFIG_DICT = config['model_configs'][MODEL_CONFIG_NAME]

if MODEL is None:
    raise Exception
elif MODEL == "HRNetV2M7":
    from ScantensusPT.nets.HRNetV2M7 import get_seg_model
elif MODEL == "HRNetV2M8":
    from ScantensusPT.nets.HRNetV2M8 import get_seg_model
elif MODEL == "HRNetV2M9":
    from ScantensusPT.nets.HRNetV2M8 import get_seg_model
elif MODEL == "HRNetV2M10":
    from ScantensusPT.nets.HRNetV2M10 import get_seg_model
else:
    raise Exception


#### inference ####
EPOCH = DATA_CONFIG_DICT['epoch']
SINGLE_INFER_BATCH_SIZE = DATA_CONFIG_DICT['single_infer_batch_size']

#### output folders ####
OUTPUT_DIR = Path(config['hosts'][HOST]['output_dir'])

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints" / PROJECT / EXPERIMENT

CHECKPOINT_KEYS_PATH = CHECKPOINT_DIR / "keys.json"
CHECKPOINT_PATH = CHECKPOINT_DIR / f'weights-{EPOCH}.pt'

#### process #####
IMAGE_LIST = DATA_CONFIG_DICT['image_list']
VALIDATION_IMAGE_FILE = DATA_DIR / "validation" / f"{IMAGE_LIST}.txt"

LABELS_TO_PROCESS = DATA_CONFIG_DICT['labels_to_process']
LABELS_TO_PROCESS_CURVE_POINTS = DATA_CONFIG_DICT['labels_to_process_curve_points']
FIREBASE_REVERSE_CONFIG_NAME = DATA_CONFIG_DICT['firebase_reverse_config_name']
FIREBASE_REVERSE_DICT = config['firebase_reverse_configs'][FIREBASE_REVERSE_CONFIG_NAME]['firebase_reverse_dict']

OUT_RUN_DIR = OUTPUT_DIR / "validation" / PROJECT / EXPERIMENT / str(EPOCH) / IMAGE_LIST
os.makedirs(OUT_RUN_DIR, exist_ok=True)

OUT_IMAGE_DIR = OUT_RUN_DIR / "images"
os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

OUT_VIDEO_DIR = OUT_RUN_DIR / "mp4"
os.makedirs(OUT_VIDEO_DIR, exist_ok=True)

OUT_LABEL_DIR = OUT_RUN_DIR / "labels"
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

OUT_FIREBASE_DIR = OUT_RUN_DIR / "firebase"
os.makedirs(OUT_FIREBASE_DIR, exist_ok=True)

OUT_CSV_DIR = OUT_RUN_DIR / "csv"
os.makedirs(OUT_CSV_DIR, exist_ok=True)

#########

def main():

    SDY_FILE = Path("/home/matthew/CMStudy_2016_12_16_064019.sdy")
    file = SDY_File(sdy_fl=SDY_FILE)
    pixel_array = file.spectrum
    pixel_array = torch.from_numpy(pixel_array.astype("float32"))

    pixel_array = torch.sqrt(pixel_array * 2) * 1.5
    pixel_array = torch.clip(pixel_array, 0, 255)
    pixel_array = torch.flip(pixel_array, [0])
    pixel_array = pixel_array.squeeze(0).squeeze(0).repeat([1,3,1,1])

    ##

    keypoint_names, keypoint_cols = get_keypoint_names_and_colors_from_json(CHECKPOINT_KEYS_PATH)

    keypoint_sd = [CURVE_SD if 'curve' in keypoint_name else DOT_SD for keypoint_name in keypoint_names]
    keypoint_sd = torch.tensor(keypoint_sd, dtype=torch.float, device=DEVICE)
    keypoint_sd = keypoint_sd.unsqueeze(1).expand(-1, 2)

    net_cfg = {}
    net_cfg['MODEL'] = {}
    net_cfg['MODEL']['PRETRAINED'] = False
    net_cfg['MODEL']['EXTRA'] = MODEL_CONFIG_DICT
    net_cfg['DATASET'] = {}
    net_cfg['DATASET']['NUM_CLASSES'] = len(keypoint_names)

    if PRE_POST:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 3 * 3
    else:
        net_cfg['DATASET']['NUM_INPUT_CHANNELS'] = 1 * 3


    single_model = get_seg_model(cfg=net_cfg)

    single_model.init_weights()
    state_dict = load_and_fix_state_dict(CHECKPOINT_PATH, device=DEVICE)
    single_model.load_state_dict(state_dict)

    print(f"Model Loading onto: {DEVICE}")

    model = single_model.to(DEVICE)

    model.eval()

    out_ys = torch.zeros(pixel_array.shape[-1], dtype=torch.float32, device="cpu")

    def get_shards(shard_width, shard_overlap, total_width):
        source_list = []
        destination_list = []

        source_start = 0
        source_end = shard_width

        destination_start = 0
        destination_end = shard_width - shard_overlap

        while destination_end <= total_width:
            source = (source_start, source_end)
            destination = (destination_start, destination_end)

            source_list.append(source)
            destination_list.append(destination)

            source_start = source_start + (shard_width - 2*shard_overlap)
            source_end = source_start + shard_width

            destination_end = source_end - shard_overlap
            destination_start = destination_end

            if source_end > total_width:
                source_end = total_width
                source_start = total_width - shard_width
                destination_end = total_width
                destination_start = total_width - shard_width + shard_overlap

                source = (source_start, source_end)
                destination = (destination_start, destination_end)

                source_list.append(source)
                destination_list.append(destination)
                break

        return source_list, destination_list

    source_list, destination_list = get_shards(1024, 256, pixel_array.shape[-1])

    for i, (source, destination) in enumerate(zip(source_list, destination_list)):
        print(i)
        source_start, source_end = source
        destination_start, destination_end = destination

        image_t = pixel_array[..., source_start:source_end]
        image_t = image_t.to(device=DEVICE, dtype=torch.float32, non_blocking=True).div(255.0).add(-0.5)

        with torch.no_grad():
            y_pred_25_clean, y_pred_50_clean = model(image_t)

            y_pred_25 = torch.nn.functional.interpolate(y_pred_25_clean, scale_factor=4, mode='bilinear', align_corners=True)
            y_pred_50 = torch.nn.functional.interpolate(y_pred_50_clean, scale_factor=2, mode='bilinear', align_corners=True)

            y_pred = (y_pred_25 + y_pred_50) / 2.0
            y_pred = gaussian_blur2d_norm(y_pred=y_pred, kernel_size=(25, 25), sigma=keypoint_sd)
            y_pred = torch.clamp(y_pred, 0, 1)

            del y_pred_25, y_pred_50

        if PRE_POST:
            image_t = image_t[:, 3:6, :, :]

        ###

        y_pred_raw = image_logit_overlay_alpha_t(logits=y_pred, images=None, cols=keypoint_cols)
        y_pred_raw = y_pred_raw.mul_(255).type(torch.uint8).cpu()

        out_path = OUT_IMAGE_DIR / "test" / "raw" / f"{i}.png"
        print(f"Saving raw: {out_path}")
        os.makedirs(Path(out_path).parent, exist_ok=True)
        torchvision.io.write_png(filename=str(out_path), input=y_pred_raw[0, ...], compression_level=7)

        del y_pred_raw

        ###

        y_pred_mix = image_logit_overlay_alpha_t(logits=y_pred, images=image_t.add(0.5), cols=keypoint_cols)
        y_pred_mix = y_pred_mix.mul_(255).type(torch.uint8).cpu()

        out_path = OUT_IMAGE_DIR / "test" / "mix" / f"{i}.png"
        print(f"Saving mix: {out_path}")
        os.makedirs(Path(out_path).parent, exist_ok=True)
        torchvision.io.write_png(filename=str(out_path), input=y_pred_mix[0, ...], compression_level=7)

        del y_pred_mix

        ###
        out_labels_dict = {}
        out_labels_dict["SDY"] = {}
        out_labels_dict["SDY"]['labels'] = {}

        label = 'curve-flow'

        ys, xs, confs = heatmap_to_label(y_pred=y_pred[0, ...],
                                         keypoint_names=keypoint_names,
                                         label=label)
        ys_all = torch.zeros(image_t.shape[-1], dtype=torch.float32, device="cpu")
        xs = torch.tensor(xs, device="cpu")
        ys = torch.tensor(ys, device="cpu")
        ys_all[xs] = ys
        out_ys[destination_start:destination_end] = ys_all[(destination_start - source_start):(destination_end - source_start)]


    print("done")



if __name__ == '__main__':
    main()