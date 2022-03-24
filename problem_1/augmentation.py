import torch
import numpy as np

# w, h, d version RandomCrop
class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant',
                           constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant',
                           constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1],
                      d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1],
                              image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label']).long(),
                'onehot_label': torch.from_numpy(sample['onehot_label']).long()
            }
        else:
            return {
                'image': torch.from_numpy(image),
                'label': torch.from_numpy(sample['label']).long()
            }


# self-written crop function
# def crop(image: ndarray):
#     crop_w, crop_h = 112, 112
#     w, h, c = image.shape

#     # if w < crop_w or h < crop_h:
#     #     image = pad(image, size=[crop_w, crop_h])
#     #     w, h, c = image.shape

#     # randomly choose cropping window
#     ww = np.random.randint(low=0, high=(w - crop_w))
#     hh = np.random.randint(low=0, high=(h - crop_h))

#     cropped_image = image[ww:ww+crop_w, hh:hh+crop_h, :]
#     return cropped_image

# RandomCrop class from pytorch tutorial
# class RandomCrop(object):
#     """Crop randomly the image in a sample.
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size

#     def __call__(self, sample):
#         image, landmarks = sample['image'], sample['landmarks']

#         w, h = image.shape[:2]
#         new_w, new_h = self.output_size

#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)

#         image = image[top:top + new_h, left:left + new_w]

#         landmarks = landmarks - [left, top]

#         return {'image': image, 'landmarks': landmarks}

# self-written flip function
# def flip(image: ndarray):
#     if np.rand() > 0.5:
#         image = image[:, ::-1, :]  # horizontal
#     if np.rand() > 0.5:
#         image = image[::-1, :, :]  # vertical
#     return image

# self-written rotate image
# def rotate(image: ndarray):
#     angle = np.rand() * pi * 2
#     image = rotate(image, angle)