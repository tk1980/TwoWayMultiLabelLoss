import torch
import torchvision.transforms as trans

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd=0.1, 
                 eigval=IMAGENET_PCA['eigval'], 
                 eigvec=IMAGENET_PCA['eigvec']):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        # Create a random vector of 3x1 in the same type as img
        alpha = img.new_empty(3,1).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .matmul(alpha * self.eigval.view(3, 1))

        return img.add(rgb.view(3, 1, 1).expand_as(img))


def get_transforms(args):
    normalize = trans.Normalize(mean=IMAGENET_STATS['mean'],
                                                 std=IMAGENET_STATS['std'])

    #- transform -#
    train_transform = trans.Compose([
                trans.RandomResizedCrop(args.image_size),
                trans.RandomHorizontalFlip(),
                trans.ColorJitter(0.4, 0.4, 0.4),
                trans.ToTensor(),
                Lighting(0.1),
                normalize
            ])

    val_transform = trans.Compose([
                trans.Resize(args.crop_size),
                trans.CenterCrop(args.image_size),
                trans.ToTensor(),
                normalize 
            ])
    return train_transform, val_transform