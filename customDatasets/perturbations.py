#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 14:49:40 2025

@author: jamie
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class GaussianPixelNoise(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns images already in [0..255] (uint8 or float).
      2) Adds Gaussian noise in integer space, clamping to [0..255].
      3) Converts back to [0..1] float before returning.
    """

    def __init__(self, base_dataset, standard_deviation=9):
        """
        :param base_dataset: A Dataset that returns (image, mask) in [0..255].
        :param standard_deviation: The std of the Gaussian noise in integer scale, 
                                   e.g. 2 => ±2 pixel intensities in [0..255].
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.std = standard_deviation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # image in [0..255], likely torch.uint8
        image_255, mask = self.base_dataset[idx]

        # Convert to float for adding noise
        float_image = image_255.float()

        noise = torch.normal(
            mean=0.0,
            std=float(self.std),
            size=float_image.shape,
            device=float_image.device,
            dtype=torch.float
        )

        noisy_float = float_image + noise
        noisy_clamped = noisy_float.clamp(0, 255).round()
        noisy_uint8 = noisy_clamped.to(torch.uint8)

        # Convert back to [0..1]
        perturbed_image = noisy_uint8.float() / 255.0

        return perturbed_image, mask


class GaussianBlur(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns images in [0..255].
      2) Repeatedly convolve with a 3×3 averaging kernel in integer space,
         exactly 'num_blur_passes' times.
      3) Clamps results to [0..255].
      4) Converts back to float [0..1] before returning.
    """

    def __init__(self, base_dataset, num_blur_passes=0):
        """
        :param base_dataset: A Dataset that returns (image, mask) in [0..255].
        :param num_blur_passes: How many times to convolve with 3×3 mask (0..9).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.num_blur_passes = num_blur_passes

        # A 3x3 averaging filter: shape [out_channels, in_channels/groups, 3, 3].
        # We'll replicate this for each color channel at runtime.
        self.kernel_3x3 = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]  # [C,H,W] in [0..255]
        
        # We'll do repeated blur in integer space, but conv2d expects float
        blurred_int = image_255  # start with the integer image

        for _ in range(self.num_blur_passes):
            float_image = blurred_int.float().unsqueeze(0)  # [1,C,H,W]
            c = float_image.shape[1]  # number of channels

            # replicate kernel for each channel
            kernel = self.kernel_3x3.to(float_image.device).expand(c, 1, 3, 3)

            # group convolution: each channel convolved separately
            convolved = F.conv2d(
                float_image,
                kernel,
                bias=None,
                stride=1,
                padding=1,  # keep same spatial size
                groups=c
            )  # [1,C,H,W] float

            convolved_clamped = convolved.round().clamp(0, 255)
            blurred_int = convolved_clamped.squeeze(0).to(torch.uint8)

        # Convert final integer image to [0..1]
        blurred_float = blurred_int.float() / 255.0

        return blurred_float, mask


class ContrastIncrease(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns [0..255].
      2) Multiplies pixel values by 'scale_factor', clamping to [0..255].
      3) Converts back to [0..1] float.
    """

    def __init__(self, base_dataset, scale_factor=1.0):
        """
        :param base_dataset: A Dataset returning [0..255].
        :param scale_factor: Factor to multiply pixel intensities (e.g. >1 for increase).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]

        scaled_float = image_255.float() * self.scale_factor
        scaled_clamped = scaled_float.round().clamp(0, 255)
        scaled_uint8 = scaled_clamped.to(torch.uint8)

        final_image = scaled_uint8.float() / 255.0
        return final_image, mask


class ContrastDecrease(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns [0..255].
      2) Multiplies each pixel by 'scale_factor' <= 1.0.
      3) Clamps to [0..255].
      4) Converts back to [0..1].
    """

    def __init__(self, base_dataset, scale_factor=1.0):
        """
        :param base_dataset: A Dataset returning [0..255].
        :param scale_factor: Factor <1.0 for contrast decrease.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]

        scaled_float = image_255.float() * self.scale_factor
        scaled_clamped = scaled_float.round().clamp(0, 255)
        scaled_uint8 = scaled_clamped.to(torch.uint8)

        final_image = scaled_uint8.float() / 255.0
        return final_image, mask


class BrightnessIncrease(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns [0..255].
      2) Adds 'offset' to each pixel, clamping to [0..255].
      3) Converts back to [0..1].
    """

    def __init__(self, base_dataset, offset=0):
        """
        :param base_dataset: A Dataset returning [0..255].
        :param offset: Amount to add to each pixel (0..45).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]

        bright_float = image_255.float() + float(self.offset)
        bright_clamped = bright_float.clamp(0, 255).round().to(torch.uint8)

        final_image = bright_clamped.float() / 255.0
        return final_image, mask


class BrightnessDecrease(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns [0..255].
      2) Subtracts 'offset' from each pixel.
      3) Clamps to [0..255].
      4) Converts back to [0..1].
    """

    def __init__(self, base_dataset, offset=0):
        """
        :param base_dataset: A Dataset returning [0..255].
        :param offset: Amount to subtract (0..45).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]

        darker_float = image_255.float() - float(self.offset)
        darker_clamped = darker_float.round().clamp(0, 255).to(torch.uint8)

        final_image = darker_clamped.float() / 255.0
        return final_image, mask


class OcclusionIncrease(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns [0..255].
      2) Randomly replaces a square region of 'square_size' with black (0).
      3) Converts back to [0..1].
    """

    def __init__(self, base_dataset, square_size=0):
        """
        :param base_dataset: A Dataset returning [0..255].
        :param square_size: side length of the occlusion square (0..45).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.square_size = square_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]  # [C,H,W] in [0..255]

        if self.square_size > 0:
            _, H, W = image_255.shape

            max_y = H - self.square_size
            max_x = W - self.square_size

            if max_y > 0 and max_x > 0:
                y0 = random.randint(0, max_y)
                x0 = random.randint(0, max_x)
                image_255[:, y0:y0+self.square_size, x0:x0+self.square_size] = 0

        final_image = image_255.float().clamp(0, 255) / 255.0
        return final_image, mask


class SaltPepperNoise(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns [0..255].
      2) Replaces a fraction 'amount' of pixels with 0 or 255.
         For each chosen pixel location, all channels are set to 0 or 255.
      3) Converts back to [0..1].
    """

    def __init__(self, base_dataset, amount=0.0):
        """
        :param base_dataset: A Dataset returning [0..255].
        :param amount: Fraction of pixels to replace with salt/pepper.
                       e.g. 0.02 => 2% of pixels
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.amount = amount

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image_255, mask = self.base_dataset[idx]  # [C,H,W] in [0..255]

        if self.amount > 0.0:
            C, H, W = image_255.shape
            total_pixels = H * W
            num_sp = int(round(self.amount * total_pixels))

            for _ in range(num_sp):
                y = random.randint(0, H - 1)
                x = random.randint(0, W - 1)
                # 50/50 salt or pepper
                if random.random() < 0.5:
                    image_255[:, y, x] = 0   # pepper
                else:
                    image_255[:, y, x] = 255 # salt

        noisy_float = image_255.float().clamp(0, 255) / 255.0
        return noisy_float, mask
