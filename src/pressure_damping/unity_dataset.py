from collections import namedtuple
import torch
from torch.utils.data import Dataset
import logging
import requests
import requests_cache
import json
import random
from src.utils.cluster_source import ClusterSource
from kornia.augmentation import RandomErasing
import datetime
import math

UnityTrainSetT = namedtuple("UnityTrainSetT", "image unity_f_code label_data label_height_shift label_width_shift transform_matrix")
CLUSTER_STORE_URL = "https://storage.googleapis.com/scantensus/fiducial"

class UnityDataset(Dataset):

    def __init__(self,
                 database_url,
                 keypoint_names,
                 transform=False,
                 transform_translate=True,
                 transform_scale=True,
                 transform_rotate=True,
                 transform_shear=True,
                 image_crop_size=(608, 608),
                 image_out_size=(512, 512),
                 pre_post=False,
                 pre_post_list=None,
                 bw_images=False,
                 device="cpu",
                 name=None):

        super().__init__()

        self.logger = logging.getLogger()
        self.keypoint_names = keypoint_names
        self.transform = transform
        self.transform_translate = transform_translate
        self.transform_rotate = transform_rotate
        self.transform_scale = transform_scale
        self.transform_shear = transform_shear
        self.image_crop_size = image_crop_size
        self.image_out_size = image_out_size
        self.pre_post = pre_post
        self.pre_post_list = pre_post_list
        self.bw_images = bw_images
        self.device = device
        self.name = name

        r = requests.get(database_url)
        if r.status_code != 200:
            raise Exception(f"Failed to load database url {database_url}")

        self.db_raw = json.loads(r.content)

        self.image_fn_list = list(self.db_raw.keys())
        logging.info(f"Number of cases {len(self.image_fn_list)}")
        hard_list = []

        if False:
            for key, data in self.db_raw.items():
                try:
                    if data['labels']['av-centre']['type'] == 'point':
                        hard_list.append(key)
                except Exception:
                    pass

            logging.info(f"Hard list has {len(hard_list)} items")

            self.image_fn_list.extend(hard_list * 6)

        self.aug_random_erase = RandomErasing(scale=(0.05, 0.15),
                                              ratio=(0.3, 3.3),
                                              value=0.0,
                                              same_on_batch=False,
                                              p=1,
                                              keepdim=False)

        self.png_session = None

    def __len__(self):
        return len(self.image_fn_list)

    def __getitem__(self, idx):

        if not self.png_session:
            self.png_session = requests_cache.CachedSession(cache_name='png_cache',
                                                            use_cache_dir=True,
                                                            cache_control=False,
                                                            expire_after=datetime.timedelta(days=300),
                                                            backend='sqlite',
                                                            stale_if_error=True,
                                                            wal=True,
                                                            timeout=30)

        image_crop_size = self.image_crop_size
        image_out_size = self.image_out_size
        transform = self.transform

        unity_code = self.image_fn_list[idx]

        image_crop_size = self.image_crop_size
        device = self.device

        transform_rand_num = random.random()

        if "clusters" in unity_code:
            unity_o = ClusterSource(unity_code=unity_code,
                                    png_cache_dir=None,
                                    server_url=CLUSTER_STORE_URL)
        else:
            unity_o = None
            logging.error(f"Unknown source for {unity_code}")

        if self.pre_post:
            pre_post_list = self.pre_post_list
        else:
            pre_post_list = [0]

        out_image = []

        out_height_shift = None
        out_width_shift = None

        for offset in pre_post_list:
            image_path = unity_o.get_frame_url(frame_offset=offset)
            try:
                image = read_image_into_t(image_path=image_path,
                                          png_session=self.png_session,
                                          device=self.device)
                image, height_shift, width_shift = center_crop_or_pad_t(image=image,
                                                                        output_size=image_crop_size,
                                                                        cval=0,
                                                                        device=device)
            except Exception:
                if offset == 0:
                    logging.exception(f"failed to load {unity_code} at offset {offset}")

                image = torch.zeros((1, image_crop_size[0], image_crop_size[1]), device=device, dtype=torch.uint8)

                height_shift = None
                width_shift = None

            if height_shift is not None and width_shift is not None:
                out_height_shift = height_shift
                out_width_shift = width_shift

            if self.bw_images:
                image = image[[0], ...]
            else:
                if image.shape[0] == 1:
                    image = torch.vstack((image, image, image))

            if transform:
                if self.pre_post:
                    if offset != 0:
                        if transform_rand_num <= 0.2:
                            image = torch.zeros_like(image)
                    elif offset == 0:
                        if 0.2 < transform_rand_num <= 0.25:
                            image = torch.zeros_like(image)

            out_image.append(image)

        image = torch.cat(out_image)

        label_data = self.db_raw[unity_code]['labels']

        if out_height_shift is None and out_width_shift is None:
            out_height_shift = 0
            out_width_shift = 0
            # Note that when label_data goes through json.dumps() it becomes the string 'null'
            label_data = None

        in_out_height_ratio = image_crop_size[0] / image_out_size[0]
        in_out_width_ratio = image_crop_size[1] / image_out_size[1]

        if transform:
            translate_h, translate_w, scale_h, scale_w, rotation_theta, shear_theta = get_random_transform_parm(translate=self.transform_translate,
                                                                                                                scale=self.transform_scale,
                                                                                                                rotate=self.transform_rotate,
                                                                                                                shear=self.transform_shear)

            transform_matrix = get_affine_matrix(tx=translate_w,
                                                 ty=translate_h,
                                                 sx=scale_w * in_out_width_ratio,
                                                 sy=scale_h * in_out_height_ratio,
                                                 rotation_theta=rotation_theta,
                                                 shear_theta=shear_theta,
                                                 device=device)

            transform_matrix_inv = transform_matrix.inverse()

        else:
            transform_matrix = get_affine_matrix(tx=0,
                                                 ty=0,
                                                 sx=in_out_width_ratio,
                                                 sy=in_out_height_ratio,
                                                 rotation_theta=0,
                                                 shear_theta=0,
                                                 device=device)

            transform_matrix_inv = transform_matrix.inverse()

        image = image.float().div(255)

        if transform:
            if 0.2 < transform_rand_num <= 0.4:
                image = self.aug_random_erase(image)

        image = transform_image(image=image,
                                transform_matrix=transform_matrix_inv,
                                out_image_size=self.image_out_size)

        if transform:
            random_gamma = math.exp(random.triangular(-0.8, 0.8))
            image = image.pow(random_gamma)

        image = image.mul(255).to(torch.uint8)

        return UnityTrainSetT(image=image,
                              unity_f_code=unity_code,
                              label_data=json.dumps(label_data),
                              label_height_shift=out_height_shift,
                              label_width_shift=out_width_shift,
                              transform_matrix=transform_matrix)


import logging
import requests_cache

import torch
import torchvision


def read_image_into_t(image_path: str,
                      png_session: requests_cache.CachedSession = None,
                      device="cpu"):

    if image_path.startswith("http://") or image_path.startswith("https://"):
        r = png_session.get(image_path)
        if r.status_code == 200:
            img_bytes = torch.frombuffer(r.content, dtype=torch.uint8)
            image = torchvision.io.decode_png(img_bytes)
            logging.debug(f"{image_path}: Successfully loaded")
        else:
            raise Exception(f"Failed to load {image_path}")
    else:
        image = torchvision.io.read_image(image_path)

    # ensure dim = 3
    if image.ndim == 2:
        image = torch.unsqueeze(image, 0)

    # remove alpha layer
    if image.shape[0] == 4:
        image = image[:3, ...]

    return image

import torch


def center_crop_or_pad_t(image: torch.tensor, output_size=(608, 608), cval=0, return_shift=True, device=None) -> tuple[torch.Tensor, float, float]:
    out_h, out_w = output_size

    ndim = image.ndim
    if ndim == 2:
        in_h, in_w = image.shape
        out_image = torch.ones((out_h, out_w), dtype=image.dtype) * cval
    elif ndim == 3:
        in_c, in_h, in_w = image.shape
        out_image = torch.ones((in_c, out_h, out_w), dtype=image.dtype) * cval
    elif ndim == 4:
        in_n, in_c, in_h, in_w = image.shape
        out_image = torch.ones((in_n, in_c, out_h, out_w), dtype=image.dtype) * cval
    else:
        raise Exception(f"Expected tensor to have ndim 2, 3, or 4")

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
        label_height_shift = out_s_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h
        label_height_shift = -in_s_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
        label_width_shift = out_s_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w
        label_width_shift = -in_s_w

    out_image[..., out_s_h:out_e_h, out_s_w:out_e_w] = image[..., in_s_h:in_e_h, in_s_w:in_e_w]

    if return_shift:
        return out_image, label_height_shift, label_width_shift

    return out_image


def patch_centre_t(image: torch.tensor, patch: torch.tensor, in_range, out_range):

    in_n, in_c, in_h, in_w = image.shape
    out_n, out_c, out_h, out_w = patch.shape

    if in_c != out_c:
        raise Exception

    if in_h <= out_h:
        in_s_h = 0
        in_e_h = in_s_h + in_h
        out_s_h = (out_h - in_h) // 2
        out_e_h = out_s_h + in_h
    else:
        in_s_h = (in_h - out_h) // 2
        in_e_h = in_s_h + out_h
        out_s_h = 0
        out_e_h = out_s_h + out_h

    if in_w <= out_w:
        in_s_w = 0
        in_e_w = in_s_w + in_w
        out_s_w = (out_w - in_w) // 2
        out_e_w = out_s_w + in_w
    else:
        in_s_w = (in_w - out_w) // 2
        in_e_w = in_s_w + out_w
        out_s_w = 0
        out_e_w = out_s_w + out_w

    image[out_range, :, in_s_h:in_e_h, in_s_w:in_e_w] = patch[in_range, :, out_s_h:out_e_h, out_s_w:out_e_w,]

    #Modify in place, so no return.
    return


import math
import random

import torch


def deg2rad(x):
    return x * math.pi / 180


def get_random_transform_parm(translate=False, scale=False, rotate=False, shear=False):

    if translate:
        translate_h = random.uniform(-0.25, 0.25)
        translate_w = random.uniform(-0.25, 0.25)
    else:
        translate_h = 0
        translate_w = 0

    if scale:
        scale_h = math.exp(random.triangular(-0.6, 0.6))
        scale_w = scale_h * math.exp(random.triangular(-0.2, 0.2))
    else:
        scale_h = 1
        scale_w = 1

    if rotate:
        rotation_deg = random.triangular(-40, 40)
    else:
        rotation_deg = 0

    rotation_theta = math.pi / 180 * rotation_deg

    if shear:
        shear_deg = random.triangular(-20, 20)
    else:
        shear_deg = 0

    shear_theta = math.pi / 180 * shear_deg

    return translate_h, translate_w, scale_h, scale_w, rotation_theta, shear_theta


def get_affine_matrix(tx=0, ty=0, sx=1, sy=1, rotation_theta=0, shear_theta=0, device="cpu"):

    tf_rotate = torch.tensor([[math.cos(rotation_theta), -math.sin(rotation_theta), 0],
                              [math.sin(rotation_theta), math.cos(rotation_theta), 0],
                              [0, 0, 1]],
                             dtype=torch.float,
                             device=device)

    tf_translate = torch.tensor([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]],
                                dtype=torch.float,
                                device=device)

    tf_scale = torch.tensor([[sx, 0, 0],
                             [0, sy, 0],
                             [0, 0, 1]],
                            dtype=torch.float,
                            device=device)

    tf_shear = torch.tensor([[1, -math.sin(shear_theta), 0],
                             [0, math.cos(shear_theta), 0],
                             [0, 0, 1]],
                            dtype=torch.float,
                            device=device)

    matrix = tf_shear @ tf_scale @ tf_rotate @ tf_translate

    return matrix


def transform_image(image: torch.Tensor, transform_matrix: torch.Tensor, out_image_size=(512,512)):

    device = image.device

    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]

    out_image_h = out_image_size[0]
    out_image_w = out_image_size[1]

    identity_grid = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32, device=device)
    intermediate_grid_shape = [batch_size, out_image_h * out_image_w, 2]

    grid = torch.nn.functional.affine_grid(identity_grid, [batch_size, 1, out_image_h, out_image_w], align_corners=False)
    grid = grid.reshape(intermediate_grid_shape)

    # For some reason it gives you w, h at the output of affine_grid. So switch here.
    grid = grid[..., [1, 0]]
    grid = apply_matrix_to_coords(transform_matrix=transform_matrix, coord=grid)
    grid = grid[..., [1, 0]]

    grid = grid.reshape([batch_size, out_image_h, out_image_w, 2])

    # There is no constant selection for padding mode - so border will have to do to weights.
    image = torch.nn.functional.grid_sample(image, grid, mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0)

    return image


def apply_matrix_to_coords(transform_matrix: torch.Tensor, coord: torch.Tensor):

    if coord.dim() == 2:
        coord = coord.unsqueeze(0)

    batch_size = coord.shape[0]

    if transform_matrix.dim() == 2:
        transform_matrix = transform_matrix.unsqueeze(0)

    if transform_matrix.size()[1:] == (3, 3):
        transform_matrix = transform_matrix[:, :2, :]

    A_batch = transform_matrix[:, :, :2]
    if A_batch.size(0) != batch_size:
        A_batch = A_batch.repeat(batch_size, 1, 1)

    B_batch = transform_matrix[:, :, 2].unsqueeze(1)

    coord = coord.bmm(A_batch.transpose(1, 2)) + B_batch.expand(coord.shape)

    return coord