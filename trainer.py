import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate_melbert
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_rels': batch[7], # adj no-reshape
                    'dep_heads': batch[8],
                    'aspect_position': batch[9],
                    'dep_dirs': batch[10]
                    }
        labels = batch[6]
    else: # bert
        if args.pure_bert:
            # input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'token_type_ids': batch[1]}
            labels = batch[6]
        elif args.frame_finder:
            # input_cat_ids
            inputs = {  'input_ids': batch[4],
                        'input_aspect_ids': batch[2],
                        'token_type_ids': batch[5],
                        'aspect_indexer': batch[3],
                        'attention_mask': batch[15]}
            labels = batch[10]
        elif args.roberta_frame or args.pure_roberta:
            inputs = {  'input_ids': batch[0],
                        'input_aspect_ids': batch[2],
                        'word_indexer': batch[1],
                        'aspect_indexer': batch[3],
                        'input_attention': batch[4],
                        'aspect_attention': batch[5],
                        'target_mask': batch[6],
                        'dep_tags': batch[7],
                        'pos_class': batch[8],
                        'text_len': batch[9],
                        'aspect_len': batch[10],
                        'dep_rels': batch[12],
                        'dep_heads': batch[13],
                        'aspect_position': batch[14],
                        'dep_dirs': batch[15],
                        'input_cat_ids': batch[16],
                        'segment_ids': batch[17],
                        'attention_mask': batch[18]}
            labels = batch[11]
        else:
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'input_aspect_ids': batch[2],
                        'word_indexer': batch[1],
                        'aspect_indexer': batch[3],
                        'input_cat_ids': batch[4],
                        'segment_ids': batch[5],
                        'dep_tags': batch[6],
                        'pos_class': batch[7],
                        'text_len': batch[8],
                        'aspect_len': batch[9],
                        'dep_rels': batch[11],
                        'dep_heads': batch[12],
                        'aspect_position': batch[13],
                        'dep_dirs': batch[14],
                        'attention_mask': batch[15],
                        'aspect_attention':batch[16],
                        'neighbor_mask':batch[17]}
            labels = batch[10]
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if args.melbert:
        return my_collate_melbert


def get_bert_optimizer(args, model, t_total, warmup_step):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total, num_cycles=args.num_cycles)
    return optimizer, scheduler


def train(args, train_dataset, model, test_dataset, val_dataset):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_step = int(args.warmup_steps * len(train_dataloader))
    
    if args.embedding_type == 'bert' or args.embedding_type == 'roberta':
        optimizer, scheduler = get_bert_optimizer(args, model, t_total, warmup_step)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total opitimization steps = %d", t_total)

    #criterion = F.cross_entropy()

    global_step = 0
    best_result = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs, labels = get_input_from_batch(args, batch)
                
            logit = model(**inputs)
            '''
            print(logit)
            print(labels)
            print('%%%%%%%%%%%%%%%')
            '''
            '''
            w = torch.tensor([0.1, 1]).type(torch.cuda.FloatTensor)
            loss = F.cross_entropy(logit, labels)
            '''
            loss_fct = nn.NLLLoss(weight=torch.Tensor([1, args.class_weight]).to(args.device))
            loss = loss_fct(logit.view(-1, args.num_classes), labels.view(-1))

            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, val_dataset, model)
                    if best_result<results['f1']:
                        best_result=results['f1']
                        results, eval_loss = evaluate(args, test_dataset, model, out_error=True)
                        all_eval_results=[results]
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results


def evaluate(args, eval_dataset, model, out_error=False):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    args.eval_batch_size = 1 if out_error else 32    
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    idxs = []
    for idx, batch in enumerate(eval_dataloader):
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)
        if int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0]) != int(labels.detach().cpu().numpy()[0]):
            idxs.append(idx)
    if out_error == True:
        print("******************error output successed************************")
        output_error(args, idxs)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results, eval_loss

def output_error(args, idxs):
    file_path = os.path.join(args.error_save_dir, 'error_idx_3.txt')
    with open(file_path, 'w') as writer:
        for idx in idxs:
            writer.write(str(idx))
            writer.write('\n')
        writer.write('total num: ')
        writer.write(str(len(idxs)))

def evaluate_badcase(args, eval_dataset, model, word_vocab):

    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)

        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            if args.embedding_type == 'bert':
                sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = args.tokenizer.decode(sent_ids)
                case['aspect'] = args.tokenizer.decode(aspect_ids)
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)
            else:
                sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = ' '.join([word_vocab['itos'][i] for i in sent_ids])
                case['aspect'] = ' '.join([word_vocab['itos'][i] for i in aspect_ids])
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def all_metrics(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    pre = precision_score(y_true=labels, y_pred=preds)
    rec = recall_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
    }

def compute_metrics(preds, labels):
    return all_metrics(preds, labels)
