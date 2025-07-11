import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size, 
    hop_size, window, pad_mode, center, ref, amin, top_db, cnn14_config, cnn6_config)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import GtzanDataset, TrainSampler, EvaluateSampler, collate_fn
from models import (FeatureTransfer_Cnn14, FeatureTransfer_Cnn6, 
    Transfer_Cnn14, Transfer_Cnn6, AffectiveCnn6, FeatureAffectiveCnn6)
from segment_evaluate import SegmentEvaluator


def train(args):

    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8
    mini_data = args.mini_data if hasattr(args, 'mini_data') else False  # Default to False for full dataset

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    
    # Use minidata_features.h5 if mini_data is True
    if mini_data:
        hdf5_path = os.path.join(workspace, 'features', 'minidata_features.h5')
    else:
        hdf5_path = os.path.join(workspace, 'features', 'features.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
         'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base), 
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    if args.model_type == 'Transfer_Cnn14':
        config = cnn14_config
        model = FeatureTransfer_Cnn14(sample_rate=sample_rate, 
            window_size=config['window_size'], hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], fmin=config['fmin'], fmax=config['fmax'], 
            classes_num=classes_num, freeze_base=True)
    elif args.model_type == 'Transfer_Cnn6':
        config = cnn6_config
        model = FeatureTransfer_Cnn6(sample_rate=sample_rate, 
            window_size=config['window_size'], hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], fmin=config['fmin'], fmax=config['fmax'], 
            classes_num=classes_num, freeze_base=freeze_base)
    elif args.model_type == 'AffectiveCnn6':
        config = cnn6_config
        model = AffectiveCnn6(sample_rate=sample_rate, 
            window_size=config['window_size'], hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], fmin=config['fmin'], fmax=config['fmax'], 
            classes_num=classes_num, freeze_visual_system=freeze_base)
    elif args.model_type == 'FeatureAffectiveCnn6':
        config = cnn6_config
        model = FeatureAffectiveCnn6(sample_rate=sample_rate, 
            window_size=config['window_size'], hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], fmin=config['fmin'], fmax=config['fmax'], 
            classes_num=classes_num, freeze_visual_system=freeze_base)
    else:
        Model = eval(args.model_type)
        model = Model(sample_rate=sample_rate, window_size=window_size, 
            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
            classes_num=classes_num, freeze_base=freeze_base)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        if args.model_type in ['AffectiveCnn6', 'FeatureAffectiveCnn6']:
            # For affective models, load pretrained weights into visual system
            model.load_visual_pretrain(pretrained_checkpoint_path)
        else:
            # For other models, use standard loading
            model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path, weights_only=False)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    
    # Ensure batch norm replacement after DataParallel wrapping for FeatureTransfer models
    if hasattr(model.module, '_replace_batch_norm'):
        model.module._replace_batch_norm()

    dataset = GtzanDataset()

    # Data generator
    train_sampler = TrainSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)

    validate_sampler = EvaluateSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Use new SegmentEvaluator instead of regular Evaluator
    evaluator = SegmentEvaluator(model=model)
    
    train_bgn_time = time.time()
    
    # Train on mini batches - cycle through data loader indefinitely
    import itertools
    train_loader_cycle = itertools.cycle(train_loader)
    
    for batch_data_dict in train_loader_cycle:

        # import crash
        # asdf
        
        # Evaluate
        if iteration % 200 == 0 and iteration > 0:
            if resume_iteration > 0 and iteration == resume_iteration:
                pass
            else:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

                train_fin_time = time.time()

                # Use segment-based evaluation
                statistics = evaluator.evaluate(validate_loader)
                logging.info('Validate accuracy: {:.3f}'.format(statistics['accuracy']))
                
                # Log confusion matrix from segment-based evaluation
                if 'confusion_matrix' in statistics:
                    logging.info('Confusion matrix:')
                    logging.info(statistics['confusion_matrix'])

                statistics_container.append(iteration, statistics, 'validate')
                statistics_container.dump()

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logging.info(
                    'Train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(train_time, validate_time))

                train_bgn_time = time.time()

        # Save model 
        if iteration % 2000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['feature']))
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key != 'audio_name':  # Skip audio_name as it's a list of strings
                batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Train
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['feature'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {
                'target': do_mixup(batch_data_dict['target'], 
                    batch_data_dict['mixup_lambda'])}
        else:
            batch_output_dict = model(batch_data_dict['feature'], None)
            batch_target_dict = {
                'target': batch_data_dict['target']}

        # loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        print(iteration, loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == stop_iteration:
            break 

        iteration += 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')