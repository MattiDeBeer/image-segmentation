import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import random
class GaussianPixelNoise(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns images as float [0..1].
      2) Converts them to integer [0..255].
      3) Adds Gaussian noise in integer space, clamping to [0..255].
      4) Converts back to [0..1] float before returning.
    """

    def __init__(self, base_dataset, standard_deviation=9):
        """
        :param base_dataset: A Dataset that returns (image, mask).
                            'image' should be float in [0..1].
        :param standard_deviation: The std of the Gaussian noise in integer scale, 
                                   e.g. 2 => ±2 pixel intensities in [0..255].
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.std = standard_deviation

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]   # image in float [0..1]

        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)  # shape [C,H,W]

        noise = torch.normal(
            mean=0.0,
            std=float(self.std),
            size=image_255.shape,
            device=image_255.device,
            dtype=torch.float
        )

        noisy_float = image_255.float() + noise  
        noisy_clamped = noisy_float.clamp(0, 255).round()
        noisy_uint8 = noisy_clamped.to(torch.uint8)   

        perturbed_image = noisy_uint8.float() / 255.0

        return perturbed_image, mask


class GaussianBlur(Dataset):
    """
    A dataset wrapper that:
      1) Assumes the base dataset returns images as float [0..1].
      2) Converts them to integer [0..255].
      3) Repeatedly convolve with a 3×3 averaging kernel in integer space,
         exactly 'num_blur_passes' times.
      4) Clamps results to [0..255].
      5) Converts back to float [0..1] before returning.

    Repeated 3×3 box blurs approximate a Gaussian blur with
    increasingly large std as 'num_blur_passes' grows.
    """

    def __init__(self, base_dataset, num_blur_passes=0):
        """
        :param base_dataset: A Dataset that returns (image, mask).
                             'image' should be float in [0..1].
        :param num_blur_passes: How many times to convolve with 3×3 mask.
                                E.g. 0..9 for your assignment.
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
        image, mask = self.base_dataset[idx]  # image in [0..1] float

        # 1) Convert [0..1] float -> [0..255] integer
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)  # [C,H,W]
        
        # We will do repeated blur in integer space, but to convolve we need float
        # so the strategy is:
        #   for pass in 1..num_blur_passes:
        #       convert int->float
        #       convolve with 3x3
        #       clamp, convert back to int
        blurred_int = image_255  # start as our initial integer image

        for _ in range(self.num_blur_passes):
            # Convert to float for convolution => shape [N=1, C, H, W]
            float_image = blurred_int.float().unsqueeze(0)  # [1,C,H,W]
            c = float_image.shape[1]  # number of channels

            # replicate kernel for each channel
            kernel = self.kernel_3x3.to(float_image.device).expand(c, 1, 3, 3)

            # group convolution: each channel is convolved separately
            convolved = F.conv2d(
                float_image,
                kernel,
                bias=None,
                stride=1,
                padding=1,  # keep same spatial size
                groups=c
            )  # [1,C,H,W] float

            # clamp to [0..255], convert back to uint8
            convolved_clamped = convolved.round().clamp(0, 255)
            blurred_int = convolved_clamped.squeeze(0).to(torch.uint8)  # [C,H,W]

        # After done N passes, convert final integer image back to [0..1] float
        blurred_float = blurred_int.float() / 255.0

        return blurred_float, mask

class ContrastIncrease(Dataset):

    def __init__(self, base_dataset, scale_factor=1.0):
        """
        :param base_dataset: A Dataset returning (image, mask) with 'image' in [0..1].
        :param scale_factor: The factor by which to multiply the pixel intensities in [0..255].
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  # image in [0..1] float

        # 1) Convert [0..1] float -> [0..255] int
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)  # shape [C,H,W]

        # 2) Multiply by scale_factor in float space
        scaled_float = image_255.float() * self.scale_factor
        
        # 3) Clamp to [0..255], round back to int
        scaled_clamped = scaled_float.round().clamp(0, 255)
        scaled_uint8 = scaled_clamped.to(torch.uint8)

        # 4) Convert back to [0..1] float
        final_image = scaled_uint8.float() / 255.0

        return final_image, mask
    

class ContrastDecrease(Dataset):
    """
    A dataset wrapper that:
      1) Takes a base dataset returning (image, mask) in float [0..1].
      2) Converts the image to [0..255] int.
      3) Multiplies each pixel by 'scale_factor' (<= 1.0) to decrease contrast.
      4) Clamps to [0..255].
      5) Converts back to [0..1] float before returning.
    """

    def __init__(self, base_dataset, scale_factor=1.0):
        """
        :param base_dataset: A Dataset returning (image, mask) with 'image' in [0..1].
        :param scale_factor: The factor by which to multiply the pixel intensities in [0..255],
                             typically < 1.0 for contrast decrease.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  # image in [0..1] float

        # 1) Convert [0..1] -> [0..255] int
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)

        # 2) Multiply by scale_factor in float
        scaled_float = image_255.float() * self.scale_factor

        # 3) Clamp to [0..255] and convert back to int
        scaled_clamped = scaled_float.round().clamp(0, 255)
        scaled_uint8 = scaled_clamped.to(torch.uint8)

        # 4) Return to [0..1] float
        final_image = scaled_uint8.float() / 255.0

        return final_image, mask

class BrightnessIncrease(Dataset):
    """
    A dataset wrapper that:
      1) Takes a base dataset returning (image, mask) in float [0..1].
      2) Converts the image to integer [0..255].
      3) Adds a brightness offset in [0..255].
      4) Clamps to [0..255].
      5) Converts back to float [0..1].
    """

    def __init__(self, base_dataset, offset=0):
        """
        :param base_dataset: Dataset returning (image, mask) in [0..1].
        :param offset: How much to add to each pixel (0..45 in your assignment).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  # in [0..1]

        # 1) Convert to [0..255] int
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)

        # 2) Add offset in float, clamp to [0..255]
        bright_float = image_255.float() + float(self.offset)
        bright_clamped = bright_float.clamp(0, 255).round().to(torch.uint8)

        # 3) Convert back to [0..1] float
        final_image = bright_clamped.float() / 255.0

        return final_image, mask

import torch
from torch.utils.data import Dataset

class BrightnessDecrease(Dataset):
    """
    A dataset wrapper that:
      1) Takes a base dataset returning (image, mask) in float [0..1].
      2) Converts the image to integer [0..255].
      3) Subtracts 'offset' from each pixel.
      4) Clamps to [0..255].
      5) Converts back to float [0..1].
    """

    def __init__(self, base_dataset, offset=0):
        """
        :param base_dataset: A Dataset returning (image, mask) in [0..1].
        :param offset: Amount to subtract from each pixel (0..45, per assignment).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  # image in [0..1] float

        # 1) Convert [0..1] -> [0..255] int
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)

        # 2) Subtract offset in float, clamp to [0..255]
        darker_float = image_255.float() - float(self.offset)
        darker_clamped = darker_float.round().clamp(0, 255).to(torch.uint8)

        # 3) Convert back to [0..1] float
        final_image = darker_clamped.float() / 255.0

        return final_image, mask



class OcclusionIncrease(Dataset):
    """
    A dataset wrapper that:
      1) Takes a base dataset returning (image, mask) in float [0..1].
      2) Converts the image to integer [0..255].
      3) Replaces a randomly placed square region of 'square_size' with black (0).
      4) Clamps to [0..255] just to be safe (though we only set to 0).
      5) Converts back to float [0..1].
    """

    def __init__(self, base_dataset, square_size=0):
        """
        :param base_dataset: A Dataset returning (image, mask) in [0..1].
        :param square_size: side length of the occlusion square in pixels (0..45).
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.square_size = square_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  # image in [0..1], shape [C,H,W]

        # 1) Convert to [0..255] integer
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)  # [C,H,W]

        # If square_size=0, do nothing (no occlusion)
        if self.square_size > 0:
            _, H, W = image_255.shape

            # 2) Randomly pick top-left corner within valid range
            #    Ensure the occlusion square fits in the image
            max_y = H - self.square_size
            max_x = W - self.square_size

            # If the square is bigger than the image, clamp or skip occlusion
            if max_y > 0 and max_x > 0:
                y0 = random.randint(0, max_y)
                x0 = random.randint(0, max_x)

                # 3) Set that region to 0 (black) in all channels
                image_255[:, y0:y0+self.square_size, x0:x0+self.square_size] = 0

        # 4) Convert back to float [0..1]
        final_image = image_255.float().clamp(0, 255) / 255.0

        return final_image, mask


class SaltPepperNoise(Dataset):
    """
    A dataset wrapper that:
      1) Takes a base dataset returning (image, mask) in float [0..1].
      2) Converts the image to integer [0..255].
      3) Replaces a fraction 'amount' of pixels with 0 or 255 (salt & pepper).
         For each chosen pixel location, the entire channel dimension is set to 0 or 255.
      4) Clamps to [0..255] (not really needed if we only set to 0 or 255).
      5) Converts back to [0..1] float.
    """

    def __init__(self, base_dataset, amount=0.0):
        """
        :param base_dataset: Dataset returning (image, mask) in float [0..1].
        :param amount: Fraction of pixels to replace with salt or pepper,
                       e.g. 0.02 => 2% of pixels. Values from {0.0, 0.02, 0.04, ... 0.18}.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.amount = amount  # fraction of pixels to corrupt

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  # image in [C,H,W], float [0..1]

        # 1) Convert to [0..255] integer
        image_255 = (image * 255.0).round().clamp(0, 255).to(torch.uint8)

        # If no noise, just return the image
        if self.amount > 0.0:
            C, H, W = image_255.shape
            total_pixels = H * W

            # Number of pixels to replace with salt/pepper
            num_sp = int(round(self.amount * total_pixels))

            # For each pixel to replace:
            for _ in range(num_sp):
                # Random location
                y = random.randint(0, H - 1)
                x = random.randint(0, W - 1)
                # Salt or pepper? 50/50
                if random.random() < 0.5:
                    image_255[:, y, x] = 0   # pepper
                else:
                    image_255[:, y, x] = 255 # salt

        # 2) Convert back to float [0..1]
        noisy_float = image_255.float().clamp(0, 255) / 255.0

        return noisy_float, mask