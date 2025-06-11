import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from monai.transforms import (
    Compose,
    RandSpatialCropd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandAffined,
    RandScaleIntensityd,
    RandGaussianNoised,
    ToTensord
)


class PairedNiftiDataset(Dataset):
    """
    Loads paired CT->MRI volumes along with a binary mask.
    Applies identical spatial augmentations to image, label, and mask.
    """
    def __init__(
        self,
        root_dir,
        patch_size=(64, 64, 64),
        mode='train',
        augment=False
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode
        self.augment = augment
        self.patient_dirs = sorted(glob(os.path.join(root_dir, '*')))

        # Define dictionary-based MONAI transforms
        if self.augment and self.mode == 'train':
            self.transform = Compose([
                RandSpatialCropd(
                    keys=['image','label','mask'],
                    roi_size=patch_size,
                    random_size=False
                ),
                RandRotated(
                    keys=['image','label','mask'],
                    range_x=10, range_y=10, range_z=0,
                    prob=0.3,
                    mode=('bilinear','nearest','nearest'),
                    padding_mode='zeros'
                ),
                RandFlipd(
                    keys=['image','label','mask'],
                    spatial_axis=0,
                    prob=0.2
                ),
                RandZoomd(
                    keys=['image','label','mask'],
                    min_zoom=0.9, max_zoom=1.1,
                    prob=0.2,
                    mode=('bilinear','nearest','nearest')
                ),
                RandAffined(
                    keys=['image','label','mask'],
                    translate_range=(-5,5),
                    prob=0.2,
                    mode=('bilinear','nearest','nearest'),
                    padding_mode='zeros'
                ),
                RandScaleIntensityd(
                    keys=['image'],
                    factors=0.1,
                    prob=0.2
                ),
                RandGaussianNoised(
                    keys=['image'],
                    std=0.01,
                    prob=0.2
                ),
                ToTensord(keys=['image','label','mask']),
            ])
        else:
            self.transform = Compose([
                RandSpatialCropd(
                    keys=['image','label','mask'],
                    roi_size=patch_size,
                    random_size=False
                ),
                ToTensord(keys=['image','label','mask']),
            ])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        ct_path   = os.path.join(patient_dir, 'ct.nii.gz')
        mri_path  = os.path.join(patient_dir, 'mr.nii.gz')
        mask_path = os.path.join(patient_dir, 'mask.nii.gz')

        ct_nii   = nib.load(ct_path)
        mri_nii  = nib.load(mri_path)
        mask_nii = nib.load(mask_path)

        ct   = ct_nii.get_fdata().astype(np.float32)
        mri  = mri_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        # --- Normalization ---
        # CT: clip HU range and scale to [0, 1]
        ct = np.clip(ct, -1000, 2000)
        ct = (ct + 1000) / 3000.0

        # MRI: z-score normalization
        mri = (mri - mri.mean()) / (mri.std() + 1e-8)

        # reshape to (C, D, H, W)
        ct   = np.expand_dims(ct,   axis=0)
        mri  = np.expand_dims(mri,  axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {
            'image': ct,
            'label': mri,
            'mask':  mask
        }
        sample = self.transform(sample)

        return sample['image'], sample['label'], sample['mask']


def get_dataloaders(
    train_dir,
    val_dir,
    test_dir,
    batch_size=1,
    patch_size=(64,64,64)
):
    train_ds = PairedNiftiDataset(
        train_dir,
        patch_size=patch_size,
        mode='train',
        augment=True
    )
    val_ds = PairedNiftiDataset(
        val_dir,
        patch_size=patch_size,
        mode='val',
        augment=False
    )
    test_ds = PairedNiftiDataset(
        test_dir,
        patch_size=patch_size,
        mode='test',
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
