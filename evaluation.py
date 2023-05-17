# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 2:44 PM 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import argparse
import logging
import utils
from model import ALBEF
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from prettytable import PrettyTable
from dataset import rerank_test_dataset, create_loader
from transformers import BertTokenizer
from train_eval import get_parameter_number, evaluation


def main(args):
    logging.info(args)
    ret_table = PrettyTable()
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    test_dataset = rerank_test_dataset(args.test_data)
    test_loader = create_loader([test_dataset], [None], batch_size=[args.batch_size],
                                num_workers=[4], is_trains = [False], collate_fns=[None])[0]
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']
    config = checkpoint['config']
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    logging.info(get_parameter_number(model))
    model.load_state_dict(state_dict, strict=False)
    logging.info('load checkpoint from %s' % args.checkpoint)
    model = model.to(args.device)
    test_top1acc, test_top3acc, test_top5acc, test_mrr5 = evaluation(model, test_loader, tokenizer, args.device)
    ret_table.field_names = ["test_top1acc", "test_top3acc", "test_top5acc", "test_mrr5"]
    ret_table.add_row([test_top1acc*100, test_top3acc*100, test_top5acc*100, test_mrr5*100])
    ret_table.float_format = ".2"
    print(ret_table)


if __name__ == '__main__':
    logging.info("Begins")
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-chinese')
    parser.add_argument('--test_data', default='bert-base-chinese')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()
    main(args)