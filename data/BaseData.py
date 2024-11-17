import copy
import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class BaseData:
    def __init__(self, args):
        self.args = args
        self.label_list = self._read_labels()
        self.id2label, self.label2id = [], {}
        self.label2task_id = {}
        self.train_data, self.val_data, self.test_data = None, None, None

    def _read_labels(self):
        """
        :return: only return the label name, in order to set label index from 0 more conveniently.
        """
        id2label = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'id2label.json')))
        return id2label

    def read_and_preprocess(self, **kwargs):
        raise NotImplementedError

    def add_labels(self, cur_labels, task_id):
        for c in cur_labels:
            if c not in self.id2label:
                self.id2label.append(c)
                self.label2id[c] = len(self.label2id)
                self.label2task_id[self.label2id[c]] = task_id

    def filter(self, labels, split='train'):
        if not isinstance(labels, list):
            labels = [labels]
        split = split.lower()
        res = []
        for label in labels:
            if split == 'train':
                if self.args.debug:
                    res += copy.deepcopy(self.train_data[label])[:10]
                else:
                    res += copy.deepcopy(self.train_data[label])
            elif split in ['dev', 'val']:
                if self.args.debug:
                    res += copy.deepcopy(self.val_data[label])[:10]
                else:
                    res += copy.deepcopy(self.val_data[label])
            elif split == 'test':
                if self.args.debug:
                    res += copy.deepcopy(self.test_data[label])[:10]
                else:
                    res += copy.deepcopy(self.test_data[label])
        for idx in range(len(res)):
            res[idx]["labels"] = self.label2id[res[idx]["labels"]]
        return res


class BaseDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cur_data = self.data[idx]
        # cur_data["idx"] = idx
        # mask_head = True if random.random() > 0.5 else False
        # input_ids, attention_mask, subject_start_pos, object_start_pos = mask_entity(cur_data["input_ids"], mask_head)
        # augment_data = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "subject_start_pos": subject_start_pos,
        #     "object_start_pos": object_start_pos,
        #     "labels": cur_data["labels"],
        #     "idx": idx
        # }
        # return [cur_data, augment_data]
        return self.data[idx]
    
    def get_random_samples_by_label(self, label, k):
        """
        Get k random samples from a specified label.
        :param label: The label to filter samples by.
        :param k: The number of random samples to retrieve.
        :return: A list of k samples that have the specified label.
        """
        # Filter data based on the specified label
        filtered_data = [item for item in self.data if item['labels'] == label]

        # If k is greater than the number of available samples, return all the samples
        if k > len(filtered_data):
            return filtered_data

        # Randomly sample k elements from the filtered data
        random_samples = random.sample(filtered_data, k)
        return random_samples


class BaseTripletDataset(Dataset):
    def __init__(self, data, cur_labels):
        
        if isinstance(data, dict):
            res = []
            for key in data.keys():
                res += data[key]
            data = res
        self.data = data
        self.cur_labels = cur_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # cur_data = self.data[idx]
        # cur_data["idx"] = idx
        # mask_head = True if random.random() > 0.5 else False
        # input_ids, attention_mask, subject_start_pos, object_start_pos = mask_entity(cur_data["input_ids"], mask_head)
        # augment_data = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "subject_start_pos": subject_start_pos,
        #     "object_start_pos": object_start_pos,
        #     "labels": cur_data["labels"],
        #     "idx": idx
        # }
        # return [cur_data, augment_data]
        return self.data[idx]
    
    def convert_into_triplets(self):
        new_data = []
        for ins in self.data:
            anchor_labels = ins['labels']
            negative_samples = []
            positive_samples = []
            for label in self.cur_labels:
                if label == anchor_labels:
                    continue
                new_negative = self.data.get_random_positive_samples_by_label(label, 1)
                new_positive = self.data.get_random_positive_samples_by_label(anchor_labels, 1)
                for ins in new_negative:
                    negative_samples.append(ins)
                for ins in new_positive:
                    positive_samples.append(ins)
            
            

            
    
    def get_random_positive_samples_by_label(self, label, k):
        """
        Get k random samples from a specified label.
        :param label: The label to filter samples by.
        :param k: The number of random samples to retrieve.
        :return: A list of k samples that have the specified label.
        """
        # Filter data based on the specified label
        filtered_data = [item for item in self.data if item['labels'] == label]

        # If k is greater than the number of available samples, return all the samples
        if k > len(filtered_data):
            return filtered_data

        # Randomly sample k elements from the filtered data
        random_samples = random.sample(filtered_data, k)
        return random_samples
    
    