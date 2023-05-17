# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 10:17 AM 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import json


class rerank_train_dataset(Dataset):
    def __init__(self, ann_file, max_words=64, ratio=1, qr=1):
        self.max_words = max_words
        self.ann = []
        for line in open(ann_file):
            data = json.loads(line)
            if qr < 1:
                data['query'] = data['query'][: int(len(data['query']) * qr)]
            self.ann.append([data['query'], data])
        if ratio < 1:
            self.ann = self.ann[: int(len(self.ann) * ratio)]

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        query, ids = self.ann[index]
        datas = {'query': query[:self.max_words], 'docs': []}
        for idx, text in zip(ids['pid'] + ids['nid'], ids['pdoc'] + ids['ndoc']):
            text = text[0]
            datas['docs'].append(text[:self.max_words])
        return datas


class rerank_test_dataset(Dataset):
    def __init__(self, ann_file, max_words=64, qr=1):
        self.max_words = max_words
        self.ann = []
        for line in open(ann_file):
            data = json.loads(line)
            if qr < 1:
                data['query'] = data['query'][: int(len(data['query']) * qr)]
            self.ann.append(data)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]
        datas = {'query': info['query'][:self.max_words], 'docs': [], 'gold_max': len(info['pid'])}
        for idx, text in zip(info['pid'] + info['nid'], info['pdoc'] + info['ndoc']):
            text = text[0]
            datas['docs'].append(text[:self.max_words])
        return datas


def create_dataset(config, ratio=1, qr=1):
    train_dataset = rerank_train_dataset(config['train_file'], ratio=ratio, qr=qr)
    logging.info('train size: {}'.format(train_dataset.__len__()))
    val_dataset = rerank_test_dataset(config['val_file'], qr=qr)
    logging.info('val size: {}'.format(val_dataset.__len__()))
    test_dataset = rerank_test_dataset(config['test_file'], qr=qr)
    logging.info('test size: {}'.format(test_dataset.__len__()))
    return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns, shuffle=None):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            if shuffle is None:
                shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders