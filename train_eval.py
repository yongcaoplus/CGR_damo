# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 9:35 AM
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import argparse
import datetime
import json
import logging
import os
import random
import time
from knockknock import slack_sender
webhook_url = "https://hooks.slack.com/services/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"
import numpy as np
import torch
import wandb
import yaml

# # If you don't want your script to sync to the cloud
# os.environ['WANDB_MODE'] = 'offline'

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
from tqdm import tqdm
from pathlib import Path
import torch.distributed as dist
from prettytable import PrettyTable
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from sklearn.metrics import ndcg_score

import utils
from model import ALBEF
from metric import MetricLogger, SmoothedValue
from dataset import create_dataset, create_sampler, create_loader


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    for i, datas in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        query = []
        doc = []
        for bs in range(len(datas['query'])):
            query.append(datas['query'][bs])
            for did in range(len(datas['docs'])):
                doc.append(datas['docs'][did][bs])
        query_input = tokenizer(query, padding='longest', return_tensors="pt").to(device)
        doc_input = tokenizer(doc, padding='longest', return_tensors="pt").to(device)
        loss, loss_constrain = model(query_input, doc_input, len(datas['query']), config['pnum'] + config['train_nnum'])
        if dist.get_rank() == 0:
            wandb.log({'train_loss': loss.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
    metric_logger.synchronize_between_processes()
    return {k: "{}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, output_dir=None, output_name=''):
    # test
    model.eval()
    if output_dir is not None:
        out = open(f'{output_dir}/{output_name}_evaluation_detail.txt', 'w')
    logging.info('Computing features for evaluation...')
    top1acc, top3acc, top5acc, total = 0, 0, 0, 0
    mrr5 = 0
    # true_relevance1, true_relevance5 = [], []
    # pred_relevance1, pred_relevance5 = [], []
    for datas in tqdm(data_loader):
        query = []
        doc = []
        group_ids = []
        gold_max = []
        for bs in range(len(datas['query'])):
            query.append(datas['query'][bs])
            for did in range(len(datas['docs'])):
                doc.append(datas['docs'][did][bs])
            group_ids.append(len(doc))
            gold_max.append(datas['gold_max'][bs])
            total += 1
        query_input = tokenizer(query, padding='longest', return_tensors="pt").to(device)
        doc_input = tokenizer(doc, padding='longest', return_tensors="pt").to(device)
        query_output = model.text_encoder(query_input.input_ids, attention_mask=query_input.attention_mask,
                                          return_dict=True)
        query_embeds = model.get_diff_embeds(query_output)
        query_feat = model.query_proj(query_embeds)
        doc_output = model.text_encoder(doc_input.input_ids, attention_mask=doc_input.attention_mask,
                                        return_dict=True)
        doc_embeds = model.get_diff_embeds(doc_output)
        doc_feat = model.doc_proj(doc_embeds)
        prev = 0
        qid = 0
        # breakpoint()
        for gid, gmax in zip(group_ids, gold_max):
            hit_top1 = False
            cur_logits = query_feat[qid].view(1, model.embed_dim).matmul(
                doc_feat[prev: gid].view(gid - prev, model.embed_dim).transpose(0, 1)
            ).view(-1)
            _, topk_ids = cur_logits.topk(5)
            topk_ids = topk_ids.tolist()
            if topk_ids[0] < gmax:
                top1acc += 1
                hit_top1 = True
            pos = 0
            pos = 0
            for idx in topk_ids[:3]:
                if idx < gmax:
                    top3acc += 1
                    # mrr3 += 1 / (pos + 1)
                    break
                pos += 1
            for idx in topk_ids[:5]:
                if idx < gmax:
                    top5acc += 1
                    mrr5 += 1 / (pos + 1)
                    break
                pos += 1
            if output_dir is not None:
                myquery = ''
                mydoc = []
                dpos = 0
                for printid in topk_ids:
                    q = query[qid]
                    d = doc[printid + prev]
                    myquery = q
                    mydoc.append('<<' + str(dpos) + '>>' + d + ':{:.2f}'.format(cur_logits[printid] * 100))
                    dpos += 1
                my_gold = doc[prev] + ':{:.2f}'.format(cur_logits[0] * 100)
                out.write(str(hit_top1) + '\t' + myquery + '\t**' + my_gold + '**\t' + '||'.join(mydoc) + '\n')
            prev = gid
            qid += 1
    return top1acc / total, top3acc / total, top5acc / total, mrr5 / total


@slack_sender(webhook_url=webhook_url, channel="model_knockknock")
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    # result print setting
    val_ret_table = PrettyTable()
    test_ret_table = PrettyTable()
    val_field_names = ["epoch", "lr", "train_loss", "val_top1acc", "val_top3acc", "val_top5acc", "val_mrr5"]
    test_field_names = ["epoch", "test_top1acc", "test_top3acc", "test_top5acc", "test_mrr5"]
    val_ret_table.field_names = val_field_names
    test_ret_table.field_names = test_field_names
    val_ret_table.float_format = ".2"
    test_ret_table.float_format = ".2"
    early_stop_thresh = config['early_stop_thresh']
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    if dist.get_rank() == 0:
        wandb.init(project="Chinese re-ranking",name=config['output_dir'].split("/")[-1], config=config)
    logging.info("Creating Chinese re-ranking dataset")
    train_dataset, val_dataset, test_dataset = create_dataset(config, ratio=args.ratio, qr=args.qr)
    logging.info("Finished reading dataset.")
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset, val_dataset, test_dataset], [True, False, False], num_tasks,
                                  global_rank)
    else:
        samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train'],
                                                                      config['batch_size_test'],
                                                                      config['batch_size_test']],
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    logging.info("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    logging.info(get_parameter_number(model))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
            if 'doc_proj' in key:
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        logging.info('load checkpoint from %s' % args.checkpoint)
        logging.info(msg)
    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = utils.create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = utils.create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    logging.info("Start training")
    start_time = time.time()
    for epoch in range(0, max_epoch):
        train_stats = {}
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config)
        val_top1acc, val_top3acc, val_top5acc, val_mrr5 = evaluation(model_without_ddp, val_loader, tokenizer, device,
                                                 output_dir=config['output_dir'], output_name=f'dev-{epoch}')
        if utils.is_main_process():
            if args.evaluate:
                log_stats = {'top1acc': val_top1acc,
                             'top3acc': val_top3acc,
                             'top5acc': val_top5acc,
                             'mrr@5': val_mrr5,
                             'epoch': epoch
                             }
                with open(os.path.join(config['output_dir'], "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'top1acc': val_top1acc,
                             'top3acc': val_top3acc,
                             'top5acc': val_top5acc,
                             'mrr@5': val_mrr5,
                             'epoch': epoch
                             }
                val_ret_table.add_row([epoch, train_stats['lr'], train_stats['loss'], val_top1acc*100, val_top3acc*100, val_top5acc*100, val_mrr5*100])
                with open(os.path.join(config['output_dir'], "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if val_top1acc > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(config['output_dir'], 'checkpoint_best.pth'))
                    best = val_top1acc
                    best_epoch = epoch
                    test_top1acc, test_top3acc, test_top5acc, test_mrr5 = evaluation(model_without_ddp, test_loader,
                                                                                      tokenizer, device,
                                                                                      output_dir=config['output_dir'],
                                                                                      output_name=f'test-{epoch}')
                    test_ret_table.add_row([epoch, test_top1acc*100, test_top3acc*100, test_top5acc*100, test_mrr5*100])
            if epoch - best_epoch >= early_stop_thresh:
                logging.info("Early stopping at epoch {}".format(epoch))
                break
            log_stats['train_loss'] = float(log_stats['train_loss'])
            if dist.get_rank() == 0:
                wandb.log(log_stats)
        if args.evaluate:
            break
        lr_scheduler.step(epoch + warmup_steps + 1)
        if args.distributed:
            dist.barrier()
        # print(val_ret_table)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))
    print(val_ret_table)
    print(test_ret_table)
    if utils.is_main_process():
        with open(os.path.join(config['output_dir'], "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)
    return {"task": config['exper_mode'], "save": config['output_dir'], "best_epoch": best_epoch, "best": best}


if __name__ == '__main__':
    logging.info("Begins")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/CGR.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-chinese')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--ratio', default=1, type=float)
    parser.add_argument('--qr', default=1, type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--output_dir', default="", type=str)
    parser.add_argument('--exper_mode', default="baseline_cls", type=str)
    parser.add_argument('--batch_size_train', default=16, type=int)
    parser.add_argument('--batch_size_test', default=16, type=int)
    parser.add_argument('--early_stop_thresh', default=2, type=int)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)
    config['text_encoder'] = args.text_encoder
    config['output_dir'] = args.output_dir
    config['exper_mode'] = args.exper_mode
    config['batch_size_train'] = args.batch_size_train
    config['batch_size_test'] = args.batch_size_test
    config['early_stop_thresh'] = args.early_stop_thresh
    config['seed'] = args.seed
    logging.info("-"*30)
    for key, value in config.items():
        if key in ['pnum', 'train_nnum', 'queue_size', 'warm_up', 'optimizer', 'schedular']:
            continue
        logging.info(f"{key}: {value}")
    logging.info("-" * 30)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
