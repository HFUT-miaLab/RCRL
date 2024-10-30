import os

import torch
from torch.utils.data import Dataset
import PIL.Image as Image
from torchvision import transforms

#===========================================获取切割后的patch
class TCGA_EGFR_patch_Dataset(Dataset):
    """
    For reading all the Patches from the a single original WSI-Datasets downloaded from MoticGallery.

        :param root: Dataset's root. E.g. r'D:\WorkGroup\st\dataset\EGFR\CFB0CC13-9FBF-406E-98E9-043BB9209C2F'.

    Examples
    --------
        dataset_dev_root = r'D:\WorkGroup\st\dataset\EGFR_DEV\train'

        dataset = MoticSlideDataset(root=dataset_dev_root, coordinate=False)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

        for i, (images, _) in enumerate(dataloader):
            print(images)
    """

    def __init__(self, root, coordinate=False, magnification=20, transform=transforms.Compose([transforms.ToTensor()])):
        self.coordinate = coordinate    #坐标

        self.slide_name = root.split("\\")[-1]

        self.image_paths = []
        self.coordinates = []
        self.transform = transform   #归一化

        if magnification == 20:
            images_root = os.path.join(root, "Medium")
        else:
            images_root = root

        for patch_image_path in os.listdir(images_root):
            patch_name = patch_image_path.split('.')[0]
            patch_coordinate = torch.Tensor([int(patch_name.split('_')[0]), int(patch_name.split('_')[-1])])

            self.image_paths.append(os.path.join(images_root, patch_image_path))
            self.coordinates.append(patch_coordinate)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if self.transform is not None:
            image = self.transform(image)

        if self.coordinate:
            coordinate = self.coordinates[index]
            return image, coordinate
        else:
            return image

#=====================================================获取特征和标签

import os
import csv

import numpy as np
import torch
import torch.utils.data as Data


def get_train_valid_names(train_valid_csv, fold=0, nclass=2):
    train_names = []
    train_labels = []
    valid_names = []
    valid_labels = []

    with open(train_valid_csv, 'r') as csv_f:
        for row in csv.reader(csv_f):
            slide_id = row[0].split('.')[0] + row[0].split('.')[1]
            cls_index = int(row[2])
            fold_idx = int(row[3])
            if fold_idx != fold:
                if nclass == 4:  # 四分类
                    train_labels.append(cls_index - 1)
                    train_names.append(slide_id)
                elif nclass == 2:
                    train_labels.append(cls_index)
                    train_names.append(slide_id)
                else:
                    raise NotImplementedError
            else:
                if nclass == 4:
                    valid_labels.append(cls_index - 1)
                    valid_names.append(slide_id)
                elif nclass == 2:
                    valid_labels.append(cls_index)
                    valid_names.append(slide_id)
                else:
                    raise NotImplementedError

    _, cls_count = np.unique(np.array(train_labels), return_counts=True)
    cls_weights = np.sum(cls_count) / cls_count
    train_weights = [cls_weights[label] for label in train_labels]

    return train_names, train_labels, valid_names, valid_labels, train_weights


def get_test_names(test_csv, nclass=2):
    test_names = []
    test_labels = []

    with open(test_csv, 'r') as csv_f:
        for row in csv.reader(csv_f):
            slide_id = row[0].split('.')[0] + row[0].split('.')[1]
            cls_index = int(row[2])

            test_names.append(slide_id)
            if nclass == 4:
                test_labels.append(cls_index - 1)
            elif nclass == 3 and cls_index != 0:
                if cls_index == 1 or cls_index == 2:
                    test_labels.append(0)
                else:
                    test_labels.append(cls_index - 2)
            elif nclass == 2:
                test_labels.append(cls_index)
            else:
                raise NotImplementedError

    return test_names, test_labels


class TCGA_EGFR_feat_label_Dataset(Data.Dataset):
    def __init__(self, image_feat_root,text_feat_root,text_tokens_feat_root,sequence_features_root, names, labels):
        self.image_feat_paths = [os.path.join(image_feat_root, name + '.pth') for name in names]
        self.text_feat_paths = [os.path.join(text_feat_root, name + '.pth') for name in names]
        self.text_tokens_feat_paths = [os.path.join(text_tokens_feat_root, name + '.pth') for name in names]
        self.sequence_feat_paths = [os.path.join(sequence_features_root, name + '.pth') for name in names]
        self.labels = labels
        assert len(self.image_feat_paths) == len(self.labels)

    def __getitem__(self, index):
        # image_feat = torch.load(self.image_feat_paths[index])
        image_feat = torch.load(self.image_feat_paths[index])['feature']
        text_feat = torch.load(self.text_feat_paths[index])
        text_tokens_feat = torch.load(self.text_tokens_feat_paths[index])
        sequence_feat = torch.load(self.sequence_feat_paths[index])
        label = self.labels[index]
        return image_feat,text_feat,text_tokens_feat, sequence_feat,label

    def __len__(self):
        return len(self.image_feat_paths)

