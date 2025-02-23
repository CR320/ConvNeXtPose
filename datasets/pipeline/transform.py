import cv2
import mmcv
import numpy as np
from .utils import get_affine_transform, warp_affine_joints


class LoadImageFromFile:
    """Loading image from file.

    Args:
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Loading image from file."""
        image_file = results['image_file']
        img = mmcv.imread(image_file, self.color_type, self.channel_order, backend='cv2')
        if img is None:
            raise ValueError('Fail to read {}'.format(image_file))
        results['image'] = img

        return results


class RandomFlip:
    """Data augmentation with random image flip for bottom-up.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        image, masks, joints = results['image'], results['masks'], results['joints']
        flip_index = results['ann_info']['flip_index']

        if np.random.random() < self.flip_prob:
            width = image.shape[1]
            image = image[:, ::-1].copy() - np.zeros_like(image)
            masks = masks[:, ::-1].copy()
            joints = joints[:, flip_index]
            joints[:, :, 0] = width - joints[:, :, 0] - 1

        results['image'], results['masks'], results['joints'] = image, masks, joints

        return results


class RandomAffine:
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to [-rotation_factor, rotation_factor]
        scale_factor (list): Scaling to [1-scale_factor, 1+scale_factor]
        scale_type: wrt ``long`` or ``short`` length of the image.
        trans_factor: Translation factor.
    """

    def __init__(self,
                 rot_factor,
                 scale_factor,
                 scale_type,
                 trans_factor):
        self.max_rotation = rot_factor
        self.min_scale = scale_factor[0]
        self.max_scale = scale_factor[1]
        self.scale_type = scale_type
        self.trans_factor = trans_factor

    def _get_scale(self, image_size, resized_size):
        w, h = image_size
        w_resized, h_resized = resized_size
        if w / w_resized < h / h_resized:
            if self.scale_type == 'long':
                w_pad = h / h_resized * w_resized
                h_pad = h
            elif self.scale_type == 'short':
                w_pad = w
                h_pad = w / w_resized * h_resized
            else:
                raise ValueError(f'Unknown scale type: {self.scale_type}')
        else:
            if self.scale_type == 'long':
                w_pad = w
                h_pad = w / w_resized * h_resized
            elif self.scale_type == 'short':
                w_pad = h / h_resized * w_resized
                h_pad = h
            else:
                raise ValueError(f'Unknown scale type: {self.scale_type}')

        scale = np.array([w_pad, h_pad], dtype=np.float32)

        return scale

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        image, masks, joints, areas, sizes = \
            results['image'], results['masks'], results['joints'], results['areas'], results['sizes']

        input_size = np.array([results['ann_info']['input_size'], results['ann_info']['input_size']])
        output_size = results['ann_info']['output_size']

        height, width = image.shape[:2]
        center = np.array([width / 2, height / 2])
        img_scale = np.array([width, height], dtype=np.float32)
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
        img_scale *= aug_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.trans_factor > 0:
            dx = np.random.randint(-self.trans_factor * img_scale[0] / 200.0,
                                   self.trans_factor * img_scale[0] / 200.0)
            dy = np.random.randint(-self.trans_factor * img_scale[1] / 200.0,
                                   self.trans_factor * img_scale[1] / 200.0)

            center[0] += dx
            center[1] += dy

        # affine trans for output
        _output_size = np.array([output_size, output_size]).astype(np.float32)
        scale = self._get_scale(img_scale, _output_size)
        mat_output = get_affine_transform(
            center=center,
            scale=scale / 200.0,
            rot=aug_rot,
            output_size=_output_size)
        masks = cv2.warpAffine((masks * 255).astype(np.uint8),
                              mat_output,
                              (int(_output_size[0]), int(_output_size[1]))) / 255
        masks = (masks > 0.5).astype(np.float32)

        joints[:, :, 0:2] = warp_affine_joints(joints[:, :, 0:2], mat_output)

        # rescale area & size
        ratio = _output_size / scale
        areas = areas * ratio[0] * ratio[1]
        sizes = sizes * ((ratio[0] * ratio[1]) ** 0.5)
        # affine trans for input
        scale = self._get_scale(img_scale, input_size)
        mat_input = get_affine_transform(
            center=center,
            scale=scale / 200.0,
            rot=aug_rot,
            output_size=input_size)
        image = cv2.warpAffine(image,
                               mat_input,
                               (int(input_size[0]), int(input_size[1])))

        results['image'], results['masks'], results['joints'], results['areas'], results['sizes'] = \
            image, masks, joints, areas, sizes

        return results


class HeatmapGenerator:
    """Generate heatmaps for bottom-up models.

    Args:
        num_joints (int): Number of keypoints
        output_size (np.ndarray): Size (w, h) of feature map
    """

    def __init__(self, output_size, num_joints):
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size], dtype=np.int32)
        self.num_joints = num_joints

    def generate_kernel(self, radius, sigma):
        size = radius * 2 + 3
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        x0, y0 = radius + 1, radius + 1

        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, joints, joints_radius):
        """Generate heatmaps."""
        total_joints = joints_radius.shape[-1]
        hms = np.zeros([total_joints, self.output_size[1], self.output_size[0]], dtype=np.float32)

        for i, joints_p in enumerate(joints):
            for idx in range(total_joints):
                if idx < self.num_joints:
                    pt = joints_p[idx]
                    x, y = int(pt[0]), int(pt[1])
                    v = pt[2]
                else:
                    v = (joints_p[:, -1] > 0).sum()
                    jt_visible = joints_p[joints_p[:, -1] > 0]
                    x = int(jt_visible[:, 0].mean()) if v > 1 else -1
                    y = int(jt_visible[:, 1].mean()) if v > 1 else -1

                if v == 0 or \
                        x < 0 or y < 0 or \
                        x >= self.output_size[0] or y >= self.output_size[1]:
                    continue

                r = joints_radius[i, idx]
                sigma = r / 3
                radius = np.round(r)

                ul = int(x - radius - 1), int(y - radius - 1)
                br = int(x + radius + 2), int(y + radius + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_size[0]) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_size[1]) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_size[0])
                aa, bb = max(0, ul[1]), min(br[1], self.output_size[1])

                g = self.generate_kernel(radius, sigma)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b, c:d])

        return hms


class OffsetGenerator:
    def __init__(self, output_size, num_joints, radius=4):
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size], dtype=np.int32)
        self.num_joints = num_joints
        self.radius = radius

    def __call__(self, joints):
        output_w, output_h = self.output_size
        offset_map = np.zeros((self.num_joints*2, output_h, output_w), dtype=np.float32)
        weight_map = np.zeros((self.num_joints*2, output_h, output_w), dtype=np.float32)
        area_map = np.zeros((output_h, output_w), dtype=np.float32)

        for person_id, joints_p in enumerate(joints):
            ct_v = (joints_p[:, -1] > 0).sum()
            if ct_v <= 1:
                continue

            jt_visible = joints_p[joints_p[:, -1] > 0]
            ct_x = int(jt_visible[:, 0].mean())
            ct_y = int(jt_visible[:, 1].mean())
            if ct_x < 0 or ct_y < 0 or \
                    ct_x >= output_w or ct_y >= output_h:
                continue

            w = np.max(jt_visible[:, 0]) - np.min(jt_visible[:, 0])
            h = np.max(jt_visible[:, 1]) - np.min(jt_visible[:, 1])
            area = w * w + h * h
            if area < 32:
                continue

            for idx, pt in enumerate(joints_p):
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= output_w or y >= output_h:
                        continue

                    start_x = max(int(ct_x - self.radius), 0)
                    start_y = max(int(ct_y - self.radius), 0)
                    end_x = min(int(ct_x + self.radius), output_w)
                    end_y = min(int(ct_y + self.radius), output_h)

                    for pos_x in range(start_x, end_x):
                        for pos_y in range(start_y, end_y):
                            offset_x = pos_x - x
                            offset_y = pos_y - y
                            if offset_map[idx*2, pos_y, pos_x] != 0 \
                                    or offset_map[idx*2+1, pos_y, pos_x] != 0:
                                if area_map[pos_y, pos_x] < area:
                                    continue
                            offset_map[idx*2, pos_y, pos_x] = offset_x
                            offset_map[idx*2+1, pos_y, pos_x] = offset_y
                            weight_map[idx*2, pos_y, pos_x] = 1. / np.sqrt(area)
                            weight_map[idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area)
                            area_map[pos_y, pos_x] = area

        return offset_map, weight_map


class RadiusEncoder:
    def __init__(self, num_joints, oks_th=0.7, fixed=False, with_center=False):
        self.num_joints = num_joints
        self.oks_th = oks_th
        self.fixed = fixed
        self.with_center = with_center

    def __call__(self, scales, sigmas):
        vars = (sigmas * 2) ** 2
        joints_radius = np.zeros([len(scales), self.num_joints], dtype=np.float32)

        if self.fixed:
            joints_radius = np.ones_like(joints_radius) * 6.0
        else:
            for i in range(len(scales)):
                for k in range(self.num_joints):
                    r_2 = 2 * (-1 * np.log(self.oks_th)) * vars[k] * scales[i]
                    r = r_2 ** 0.5
                    joints_radius[i, k] = r

        if self.with_center:
            center_radius = joints_radius.mean(axis=-1, keepdims=True) * 2
            joints_radius = np.concatenate([joints_radius, center_radius], axis=-1)

        return joints_radius


class JointsEncoder:
    """Encodes the visible joints into (coordinates, score); The coordinate of
    one joint and its score are of `int` type.

    (idx * output_size**2 + y * output_size + x, 1) or (0, 0).

    Args:
        max_num_people(int): Max number of people in an image
        num_joints(int): Number of keypoints
        output_size(np.ndarray): Size (w, h) of feature map
        tag_per_joint(bool):  Option to use one tag map per joint.
    """

    def __init__(self, output_size, num_joints, max_num_people):
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size], dtype=np.int32)
        self.max_size = (self.output_size[0] * self.output_size[1]) ** 0.5
        self.num_joints = num_joints
        self.max_num_people = max_num_people

    def __call__(self, joints, areas, sizes, joints_radius):
        """
        Note:
            - number of people in image: N
            - number of keypoints: K
            - max number of people in an image: M

        Args:
            joints (np.ndarray[N,K,3])

        Returns:
            visible_kpts (np.ndarray[M,K,2]).
        """
        visible_joints = np.zeros([self.max_num_people, self.num_joints, 3], dtype=np.float32)
        valid_areas = np.zeros([self.max_num_people], dtype=np.float32)
        valid_sizes = np.zeros([self.max_num_people], dtype=np.float32)

        tot = 0
        for i, joints_p in enumerate(joints):
            num_visible = 0
            for idx, pt in enumerate(joints_p):
                x, y = int(pt[0]), int(pt[1])
                if (pt[2] > 0 and 0 <= y < self.output_size[1]
                        and 0 <= x < self.output_size[0]):
                    visible_joints[tot, idx] = (pt[0], pt[1], 1)
                    num_visible += 1

            if num_visible > 0:
                valid_areas[tot] = areas[i]
                valid_sizes[tot] = sizes[i] if sizes[i] < self.max_size else self.max_size
                tot += 1

        return visible_joints, valid_areas, valid_sizes


class FormatGroundTruth:
    """Generate multi-scale heatmap target for associate embedding.

    Args:
        max_num_people (int): Maximum number of people in an image
    """

    def __init__(self, max_num_people, fixed_radius=False, with_center=False):
        self.max_num_people = max_num_people
        self.fixed_radius = fixed_radius
        self.with_center = with_center

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        num_joints, output_size, input_size, sigmas = \
            results['ann_info']['num_joints'], results['ann_info']['output_size'], \
            results['ann_info']['input_size'], results['ann_info']['sigmas']
        joints, areas, sizes = results['joints'], results['areas'], results['sizes']
        scales = areas * 0.53 if results['ann_info']['dataset_name'] == 'crowdpose' else areas

        radius_encoder = RadiusEncoder(num_joints, fixed=self.fixed_radius, with_center=self.with_center)
        joints_encoder = JointsEncoder(output_size, num_joints, self.max_num_people)
        heatmap_generator = HeatmapGenerator(output_size, num_joints)
        offset_generator = OffsetGenerator(output_size, num_joints)

        joints_radius = radius_encoder(scales, sigmas)
        results['target_heatmaps'] = heatmap_generator(joints, joints_radius)
        results['target_offsets'], results['offset_weights'] = offset_generator(joints)
        results['target_joints'], \
        results['target_areas'], \
        results['target_sizes'] = joints_encoder(joints, areas, sizes, joints_radius)

        return results


class ResizeAlign:
    """Align transform for bottom-up.
        base_size (int): base size
        size_divisor (int): size_divisor
    """

    def __init__(self, size_divisor, test_scale_factors):
        self.size_divisor = size_divisor
        self.test_scale_factors = test_scale_factors

    def _ceil_to_multiples_of(self, x):
        """Transform x to the integral multiple of the base."""
        return int(np.ceil(x / self.size_divisor)) * self.size_divisor

    def _get_image_size(self, image_shape, input_size):
        # calculate the size for min_scale
        h, w = image_shape
        input_size = self._ceil_to_multiples_of(input_size)

        if w < h:
            w_resized = int(input_size)
            h_resized = int(self._ceil_to_multiples_of(input_size / w * h))
            scale_w = w / 200
            scale_h = h_resized / w_resized * w / 200
        else:
            h_resized = int(input_size)
            w_resized = int(self._ceil_to_multiples_of(input_size / h * w))
            scale_h = h / 200
            scale_w = w_resized / h_resized * h / 200

        base_size = (w_resized, h_resized)
        center = [round(w / 2.0), round(h / 2.0)]
        scale = [scale_w, scale_h]

        return base_size, center, scale

    def __call__(self, results):
        """Resize multi-scale size and align transform for bottom-up."""
        input_size = results['ann_info']['input_size']
        image = results['image']

        # get base_size, center & scale info for input image
        base_size, center, scale = self._get_image_size(image.shape[0:2], input_size)
        results['base_size'], results['center'], results['scale'] = base_size, center, scale

        # multi-scale resize
        assert self.test_scale_factors[0] == 1
        resized_images = list()
        for scale_factor in self.test_scale_factors:
            scaled_size = (int(base_size[0] * scale_factor), int(base_size[1] * scale_factor))
            trans = get_affine_transform(np.array(center), np.array(scale), 0, scaled_size)
            resized_images.append(cv2.warpAffine(image, trans, scaled_size))

        results['image'] = resized_images

        return results


class NormalizeImage:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        images = results['image']

        if isinstance(images, np.ndarray):
            images = mmcv.imnormalize(images, self.mean, self.std, self.to_rgb)
            norm_images = images.transpose(2, 0, 1)
        elif isinstance(images, list):
            norm_images = list()
            for image in images:
                image = mmcv.imnormalize(image, self.mean, self.std, self.to_rgb)
                norm_images.append(image.transpose(2, 0, 1))
        else:
            raise TypeError('Unsupported image type:{}'.format(type(images)))

        results['image'] = norm_images

        return results
