import os
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from data.data_loader import get_datasets, get_dataloader
from eval import eval_model
from models.architecture import PipeLine

from utils.loss_func import OverallLoss
from utils.evaluation_metric import matching_accuracy, compute_metrics, summarize_metrics, print_metrics
from utils.hungarian import hungarian
from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.print_easydict import print_easydict

writer = None


def train_eval_model(model, overallLoss, optimizer, dataloader, num_epochs=200, resume=False, start_epoch=0):
    print('**************************************')
    print('Start training...')
    dataset_size = len(dataloader['train'].dataset)
    print('train datasize: {}'.format(dataset_size))

    since = time.time()
    lap_solver = hungarian
    optimal_acc = 0.0
    optimal_rot = np.inf
    device = next(model.parameters()).device

    print('model on device: {}'.format(device))

    checkpoint_path = Path(cfg.OUTPUT_PATH) / 'checkpoints'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    if resume:
        assert start_epoch != 0
        model_path = str(checkpoint_path / 'model_{:04}.pth'.format(start_epoch))
        print('Loading model parameters from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['params'])
        optimizer.load_state_dict(checkpoint['optim'])
        assert checkpoint['epoch'] == start_epoch
        print('Current epoch: {}'.format(checkpoint['epoch']))
        print('Current loss: {}'.format(checkpoint['loss']))

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=cfg.TRAIN.LR_STEP,
                                               gamma=cfg.TRAIN.LR_DECAY,
                                               last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Set model to training mode
        model.train()

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        iter_num = 0
        all_train_metrics_np = defaultdict(list)

        for inputs in dataloader['train']:
            points_src, points_ref = [_.cuda() for _ in inputs['points']]
            num_src, num_ref = [_.cuda() for _ in inputs['num']]
            perm_mat = inputs['perm_mat_gt'].cuda()
            transform_gt, _ = [_.cuda() for _ in inputs['transform_gt']]
            src_overlap_gt, ref_overlap_gt = [_.cuda() for _ in inputs['overlap_gt']]
            points_src_raw = inputs['points_src_raw'].cuda()
            points_ref_raw = inputs['points_ref_raw'].cuda()

            batch_cur_size = perm_mat.size(0)
            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                data_dict = model(points_src, points_ref, 'train')

                overlap_pred = torch.cat((data_dict['src_overlap'], data_dict['ref_overlap']), dim=1)
                overlap_gt = torch.cat((src_overlap_gt, ref_overlap_gt), dim=1)

                loss_item = overallLoss(data_dict['s_pred'], perm_mat, num_src, num_ref,
                                        overlap_pred, overlap_gt, points_src_raw, points_ref_raw,
                                        data_dict['coarse_src'], data_dict['fine_src'],
                                        data_dict['coarse_ref'], data_dict['fine_ref'], data_dict['prob'])
                loss = loss_item['perm_loss'] + loss_item['overlap_loss'] \
                       + 1 * loss_item['c_s_cd_loss'] + 1 * loss_item['c_r_cd_loss'] \
                       + 1 * loss_item['f_s_cd_loss'] + 1 * loss_item['f_r_cd_loss'] \
                       + 0.5 * loss_item['overlap_prob_loss'] + 0.1 * loss_item['kl_loss']

                # backward + optimize
                loss.backward()
                optimizer.step()

                # training accuracy statistic
                s_perm_mat = lap_solver(data_dict['s_pred'], num_src, num_ref,
                                        data_dict['src_row_sum'], data_dict['ref_col_sum'])
                match_metrics = matching_accuracy(s_perm_mat, perm_mat, num_src)
                perform_metrics = compute_metrics(s_perm_mat, points_src[:, :, :3], points_ref[:, :, :3],
                                                  transform_gt[:, :3, :3], transform_gt[:, :3, 3],
                                                  data_dict['src_overlap'], data_dict['ref_overlap'])

                for k in match_metrics:
                    all_train_metrics_np[k].append(match_metrics[k])
                for k in perform_metrics:
                    all_train_metrics_np[k].append(perform_metrics[k])
                all_train_metrics_np['perm_loss'].append(np.repeat(loss_item['perm_loss'].item(), batch_cur_size))
                all_train_metrics_np['overlap_loss'].append(np.repeat(loss_item['overlap_loss'].item(), batch_cur_size))
                all_train_metrics_np['c_s_cd_loss'].append(np.repeat(loss_item['c_s_cd_loss'].item(), batch_cur_size))
                all_train_metrics_np['f_s_cd_loss'].append(np.repeat(loss_item['f_s_cd_loss'].item(), batch_cur_size))
                all_train_metrics_np['c_r_cd_loss'].append(np.repeat(loss_item['c_r_cd_loss'].item(), batch_cur_size))
                all_train_metrics_np['f_r_cd_loss'].append(np.repeat(loss_item['f_r_cd_loss'].item(), batch_cur_size))
                all_train_metrics_np['overlap_prob_loss'].append(
                    np.repeat(loss_item['overlap_prob_loss'].item(), batch_cur_size))
                all_train_metrics_np['kl_loss'].append(np.repeat(loss_item['kl_loss'].item(), batch_cur_size))

                if iter_num % cfg.STATISTIC_STEP == 0:
                    iter_log = '[{}] Epoch: [{:<3}/{:<3}] || Iteration: [{:<4}/{:<4}]' \
                        .format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), epoch, num_epochs - 1, iter_num,
                                len(dataloader['train']))
                    for k in all_train_metrics_np:
                        if k.endswith('loss') or k.startswith('acc'):
                            iter_log += ' || ' + k + ': {:.4f}'.format(
                                np.mean(np.concatenate(all_train_metrics_np[k])[-cfg.STATISTIC_STEP * batch_cur_size:]))
                    print(iter_log)

        all_train_metrics_np = {k: np.concatenate(all_train_metrics_np[k]) for k in all_train_metrics_np}
        summary_metrics = summarize_metrics(all_train_metrics_np)

        epoch_log = 'Epoch: {:<4}'.format(epoch)
        for k in summary_metrics:
            if k.endswith('loss') or k.startswith('acc'):
                epoch_log += ' Mean-' + k + ': {:.4f}'.format(summary_metrics[k])
        print(epoch_log)

        print_metrics(summary_metrics)

        save_path = str(checkpoint_path / 'model_{:04}.pth'.format(epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'params': model.state_dict(),
            'optim': optimizer.state_dict(),
            'loss': loss
        }, save_path)

        for k in summary_metrics:
            if k.endswith('loss'):
                writer.add_scalar('train-loss/' + k, summary_metrics[k], epoch)
        writer.add_scalar('train/acc', summary_metrics['acc_gt'], epoch)
        writer.add_scalar('train/r_rmse', summary_metrics['r_rmse'], epoch)
        writer.add_scalar('train/t_rmse', summary_metrics['t_rmse'], epoch)
        writer.add_scalar('train/r_mae', summary_metrics['r_mae'], epoch)
        writer.add_scalar('train/t_mae', summary_metrics['t_mae'], epoch)
        writer.add_scalar('train/err_r_deg_mean', summary_metrics['err_r_deg_mean'], epoch)
        writer.add_scalar('train/err_t_mean', summary_metrics['err_t_mean'], epoch)

        # Eval in each epoch
        val_metrics = eval_model(model, dataloader['val'])

        writer.add_scalar('val/acc', val_metrics['acc_gt'], epoch)
        writer.add_scalar('val/r_rmse', val_metrics['r_rmse'], epoch)
        writer.add_scalar('val/t_rmse', val_metrics['t_rmse'], epoch)
        writer.add_scalar('val/r_mae', val_metrics['r_mae'], epoch)
        writer.add_scalar('val/t_mae', val_metrics['t_mae'], epoch)
        writer.add_scalar('val/err_r_deg_mean', val_metrics['err_r_deg_mean'], epoch)
        writer.add_scalar('val/err_t_mean', val_metrics['err_t_mean'], epoch)

        if optimal_acc < val_metrics['acc_gt']:
            optimal_acc = val_metrics['acc_gt']
            save_best_acc_pth = str(checkpoint_path / 'model_best_acc.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'params': model.state_dict(),
                'optim': optimizer.state_dict(),
                'loss': loss
            }, save_best_acc_pth)
            print('Current best acc model is {}'.format(epoch + 1))
        if optimal_rot > val_metrics['err_r_deg_mean']:
            optimal_rot = val_metrics['err_r_deg_mean']
            save_best_path = str(checkpoint_path / 'model_best.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'params': model.state_dict(),
                'optim': optimizer.state_dict(),
                'loss': loss
            }, save_best_path)
            print('Current best rotation error model is {}'.format(epoch + 1))

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point could registration training & evaluation code.')
    parser.add_argument('--cfg', dest='cfg_file', help='an optional config file',
                        default='experiments/UTOPIC_Unseen_CropRPM_0.7_modelnet40.yaml', type=str)

    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        out_path = get_output_dir(cfg.MODEL_NAME,
                                  cfg.DATASET_NAME + ('_Unseen_' if cfg.DATASET.UNSEEN else '_Seen_') +
                                  cfg.DATASET.NOISE_TYPE + ('_' + str(cfg.DATASET.PARTIAL_P_KEEP[0])))
        cfg_from_list(['OUTPUT_PATH', out_path])
    assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    writer = SummaryWriter(log_dir=os.path.join(str(Path(cfg.OUTPUT_PATH)), 'tensorboard'))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)

    torch.manual_seed(cfg.RANDOM_SEED)

    pc_dataset = {}
    pc_dataset['train'] = get_datasets(partition='train',
                                       num_points=cfg.DATASET.POINT_NUM,
                                       unseen=cfg.DATASET.UNSEEN,
                                       noise_type=cfg.DATASET.NOISE_TYPE,
                                       rot_mag=cfg.DATASET.ROT_MAG,
                                       trans_mag=cfg.DATASET.TRANS_MAG,
                                       partial_p_keep=cfg.DATASET.PARTIAL_P_KEEP,
                                       crossval=True,
                                       train_part=True)
    pc_dataset['val'] = get_datasets(partition='train',
                                     num_points=cfg.DATASET.POINT_NUM,
                                     unseen=cfg.DATASET.UNSEEN,
                                     noise_type=cfg.DATASET.NOISE_TYPE,
                                     rot_mag=cfg.DATASET.ROT_MAG,
                                     trans_mag=cfg.DATASET.TRANS_MAG,
                                     partial_p_keep=cfg.DATASET.PARTIAL_P_KEEP,
                                     crossval=True,
                                     train_part=False)

    dataloader = {x: get_dataloader(pc_dataset[x], shuffle=(x == 'train'), phase=x) for x in ('train', 'val')}

    model = PipeLine()
    model = model.cuda()
    overallLoss = OverallLoss()

    if cfg.TRAIN.OPTIM == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('train_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        model = train_eval_model(model, overallLoss, optimizer, dataloader,
                                 num_epochs=cfg.TRAIN.NUM_EPOCHS,
                                 resume=cfg.TRAIN.START_EPOCH != 0,
                                 start_epoch=cfg.TRAIN.START_EPOCH)
