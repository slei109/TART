import datetime

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from dataset.sampler import ParallelSampler_Test


def test_one(task, model, args):
    '''
        Evaluate the model on one sampled task. Return the accuracy.
    '''
    support, query = task

    if args.embedding != 'bilstm':

        # Embedding the document
        XS = model['G'](support)
        YS = support['label']

        XQ = model['G'](query)
        YQ = query['label']

        # Apply the classifier
        acc, _, _ = model['clf'](XS, YS, XQ, YQ)

        return acc

    else:
        # Embedding the document
        XS, XS_inputD, XS_avg, XS_loss, XS_w2v = model['G'](support)
        YS = support['label']

        XQ, XQ_inputD, XQ_avg, XQ_loss, XQ_w2v = model['G'](query)
        YQ = query['label']

        query_data = query['text']
        if query_data.shape[1] < 50:
            zero = torch.zeros((query_data.shape[0], 50 - query_data.shape[1]))
            if args.cuda != -1:
                zero = zero.cuda(args.cuda)
            query_data = torch.cat((query_data, zero), dim=-1)
        else:
            query_data = query_data[:, :50]


        # Apply the classifier
        acc, loss, x_hat = model['clf'](XS, YS, XQ, YQ, query_data)
        all_sentence_ebd = XQ
        all_avg_sentence_ebd = XQ_avg
        all_label = YQ

        return acc, all_sentence_ebd.cpu().detach().numpy(), all_avg_sentence_ebd.cpu().detach().numpy(), all_label.cpu().detach().numpy(), query_data.cpu().detach().numpy(), x_hat.cpu().detach().numpy()


def test(test_data, model, args, num_episodes, verbose=True, sampled_tasks=None):
    '''
        Evaluate the model on a bag of sampled tasks. Return the mean accuracy
        and its std.
    '''
    model['G'].eval()
    model['clf'].eval()

    if sampled_tasks is None:
        sampled_tasks = ParallelSampler_Test(test_data, args,
                                        num_episodes).get_epoch()

    acc = []
    all_sentence_ebd = None
    all_avg_sentence_ebd = None
    all_sentence_label = None
    all_word_weight = None
    all_query_data = None
    all_x_hat = None
    all_drawn_data = {}
    if not args.notqdm:
        sampled_tasks = tqdm(sampled_tasks, total=num_episodes, ncols=80,
                             leave=False,
                             desc=colored('Testing on val', 'yellow'))
    count = 0
    for task in sampled_tasks:
        if args.embedding == 'bilstm':
            acc1, sentence_ebd, avg_sentence_ebd, sentence_label, query_data, x_hat = test_one(task, model, args)
            if count < 20:
                if all_sentence_ebd is None:
                    all_sentence_ebd = sentence_ebd
                    all_avg_sentence_ebd = avg_sentence_ebd
                    all_sentence_label = sentence_label
                    all_query_data = query_data
                    all_x_hat = x_hat
                else:
                    all_sentence_ebd = np.concatenate((all_sentence_ebd, sentence_ebd), 0)
                    all_avg_sentence_ebd = np.concatenate((all_avg_sentence_ebd, avg_sentence_ebd), 0)
                    all_sentence_label = np.concatenate((all_sentence_label, sentence_label))
                    all_query_data = np.concatenate((all_query_data, query_data), 0)
                    all_x_hat = np.concatenate((all_x_hat, x_hat), 0)
            count = count + 1
            acc.append(acc1)
        else:
            acc.append(test_one(task, model, args))

    acc = np.array(acc)

    if verbose:
        if args.embedding != 'bilstm':
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
                ), flush=True)
        else:
            print("{}, {:s} {:>7.4f}, {:s} {:>7.4f}".format(
                datetime.datetime.now(),
                colored("test acc mean", "blue"),
                np.mean(acc),
                colored("test std", "blue"),
                np.std(acc),
            ), flush=True)

    return np.mean(acc), np.std(acc), all_drawn_data