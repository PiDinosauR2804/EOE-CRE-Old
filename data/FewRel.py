import copy
import json
import os
import random
import numpy as np

from tqdm import tqdm

from .BaseData import BaseData


class FewRelData(BaseData):
    def __init__(self, args):
        super().__init__(args)
        self.entity_markers = ["[E11]", "[E12]", "[E21]", "[E22]"]
        self.eoeid2waveid = {}  

    def remove_entity_markers(self, input_ids):
        ans = []
        entity_pos = {}
        for c in input_ids:
            if c not in [30522, 30523, 30524, 30525]:
                ans.append(c)
            else:
                if c % 2 == 0:
                    entity_pos[c] = len(ans)
                else:
                    entity_pos[c] = len(ans) - 1
        return ans, entity_pos[30522], entity_pos[30523], entity_pos[30524], entity_pos[30525]

    def preprocess(self, raw_data, tokenizer):
        subject_start_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[0])
        object_start_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[2])
        subject_end_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[1])
        object_end_marker = tokenizer.convert_tokens_to_ids(self.entity_markers[3])
        res = []
        result = tokenizer(raw_data['sentence'])
        for idx in range(len(raw_data['sentence'])):
            subject_marker_st = result['input_ids'][idx].index(subject_start_marker)
            object_marker_st = result['input_ids'][idx].index(object_start_marker)
            subject_marker_ed = result['input_ids'][idx].index(subject_end_marker)
            object_marker_ed = result['input_ids'][idx].index(object_end_marker)
            input_ids = result['input_ids'][idx]
            sentence = copy.deepcopy(raw_data['sentence'][idx])
            for c in self.entity_markers:
                sentence = sentence.replace(c, '')
            sentence = sentence.replace('  ', ' ')
            # prompt_input_ids, mask_pos = self.get_prompt_input_ids(input_ids)
            input_ids_without_marker, subject_st, subject_ed, object_st, object_ed = \
                self.remove_entity_markers(input_ids)
            ins = {
                'sentence': sentence,
                'input_ids': input_ids,  # default: add marker to the head entity and tail entity
                'subject_marker_st': subject_marker_st,
                'object_marker_st': object_marker_st,
                'labels': raw_data['labels'][idx],
                'input_ids_without_marker': input_ids_without_marker,
                'subject_st': subject_st,
                'subject_ed': subject_ed,
                'object_st': object_st,
                'object_ed': object_ed,
            }
            if hasattr(self.args, 'columns'):
                columns = self.args.columns
                ins = {k: v for k, v in ins.items() if k in columns}
            res.append(ins)
        return res

    def read_and_preprocess(self, tokenizer, seed=None):
        raw_data = json.load(open(os.path.join(self.args.data_path, self.args.dataset_name, 'data_with_marker.json')))

        train_data = {}
        val_data = {}
        test_data = {}

        if seed is not None:
            random.seed(seed)

        for label in tqdm(raw_data.keys(), desc="Load FewRel data"):
            cur_data = raw_data[label]
            random.shuffle(cur_data)
            
            self.eoeid2waveid = {0: 26, 1: 15, 2: 11, 3: 58, 4: 75, 5: 21, 6: 64, 7: 53, 8: 72, 9: 67, 10: 3, 11: 17, 12: 52, 13: 63, 14: 40, 15: 39, 16: 5, 17: 47, 18: 59, 19: 2, 20: 66, 21: 65, 22: 4, 23: 43, 24: 7, 25: 42, 26: 25, 27: 16, 28: 49, 29: 54, 30: 36, 31: 76, 32: 14, 33: 46, 34: 70, 35: 77, 36: 31, 37: 69, 38: 51, 39: 13, 40: 71, 41: 35, 42: 44, 43: 62, 44: 1, 45: 61, 46: 0, 47: 24, 48: 33, 49: 37, 50: 48, 51: 79, 52: 56, 53: 41, 54: 38, 55: 20, 56: 74, 57: 34, 58: 8, 59: 12, 60: 73, 61: 6, 62: 55, 63: 18, 64: 22, 65: 45, 66: 9, 67: 30, 68: 23, 69: 78, 70: 57, 71: 50, 72: 27, 73: 68, 74: 19, 75: 28, 76: 10, 77: 60, 78: 29, 79: 32}
            
            train_raw_data = {"sentence": [], "labels": []}
            val_raw_data = {"sentence": [], "labels": []}
            test_raw_data = {"sentence": [], "labels": []}
            for idx, sample in enumerate(cur_data):
                sample["tokens"] = ' '.join(sample["tokens"])
                sample["relation"] = sample["relation"]
                if idx < 420:
                    train_raw_data["sentence"].append(sample["tokens"])
                    train_raw_data["labels"].append(sample["relation"])
                elif idx < 420 + 140:
                    val_raw_data["sentence"].append(sample["tokens"])
                    val_raw_data["labels"].append(sample["relation"])
                else:
                    test_raw_data["sentence"].append(sample["tokens"])
                    test_raw_data["labels"].append(sample["relation"])

            train_data[label] = self.preprocess(train_raw_data, tokenizer)
            val_data[label] = self.preprocess(val_raw_data, tokenizer)
            test_data[label] = self.preprocess(test_raw_data, tokenizer)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

