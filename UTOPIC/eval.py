import os
import time
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch

from data.data_loader import get_dataloader, get_datasets
from models.architecture import PipeLine

from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from utils.hungarian import hungarian
from utils.loss_func import OverallLoss
from utils.evaluation_metric import matching_accuracy,  summarize_metrics, print_metrics, compute_metrics
from utils.dup_stdout_manager import DupStdoutFileManager
from utils.print_easydict import print_easydict


def eval_model(model, dataloader, eval_epoch=None, metric_is_save=False, save_filetime='time'):
    print('-----------------Start evaluation-----------------')
    lap_solver = hungarian
    overallLoss = OverallLoss()
    since = time.time()
    all_val_metrics_np = defaultdict(list)
    iter_num = 0

    dataset_size = len(dataloader.dataset)
    print('datasize: {}'.format(dataset_size))
    device = next(model.parameters()).device
    print('model on device: {}'.format(device))

    if eval_epoch is not None:
        if eval_epoch == -1:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'checkpoints' / 'model_best.pth')
            print('Loading best model parameters')
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['params'])
            print('Best epoch: {}'.format(checkpoint['epoch']))
        else:
            model_path = str(Path(cfg.OUTPUT_PATH) / 'checkpoints' / 'model_{:04}.pth'.format(eval_epoch))
            print('Loading model parameters from {}'.format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['params'])
            assert checkpoint['epoch'] == eval_epoch
            print('Current epoch: {}'.format(checkpoint['epoch']))

    model.eval()

    for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        points_src, points_ref = [_.cuda() for _ in inputs['points']]
        num_src, num_ref = [_.cuda() for _ in inputs['num']]
        perm_mat = inputs['perm_mat_gt'].cuda()
        transform_gt, _ = [_.cuda() for _ in inputs['transform_gt']]
        src_overlap_gt, ref_overlap_gt = [_.cuda() for _ in inputs['overlap_gt']]
        points_src_raw = inputs['points_src_raw'].cuda()
        points_ref_raw = inputs['points_ref_raw'].cuda()
        Label = torch.tensor([_ for _ in inputs['label']])

        batch_cur_size = perm_mat.size(0)
        iter_num = iter_num + 1
        infer_time = time.time()

        with torch.set_grad_enabled(False):
            data_dict = model(points_src, points_ref, 'eval')

            overlap_pred = torch.cat((data_dict['src_overlap'], data_dict['ref_overlap']), dim=1)
            overlap_gt = torch.cat((src_overlap_gt, ref_overlap_gt), dim=1)

            loss_item = overallLoss(data_dict['s_pred'], perm_mat, num_src, num_ref,
                                    overlap_pred, overlap_gt, points_src_raw, points_ref_raw,
                                    data_dict['coarse_src'], data_dict['fine_src'],
                                    data_dict['coarse_ref'], data_dict['fine_ref'], data_dict['prob'])

            s_perm_mat = lap_solver(data_dict['s_pred'], num_src, num_ref,
                                    data_dict['src_row_sum'], data_dict['ref_col_sum'])

        infer_time = time.time() - infer_time
        match_metrics = matching_accuracy(s_perm_mat, perm_mat, num_src)
        perform_metrics = compute_metrics(s_perm_mat, points_src[:, :, :3], points_ref[:, :, :3],
                                          transform_gt[:, :3, :3], transform_gt[:, :3, 3],
                                          data_dict['src_overlap'], data_dict['ref_overlap'])

        for k in match_metrics:
            all_val_metrics_np[k].append(match_metrics[k])
        for k in perform_metrics:
            all_val_metrics_np[k].append(perform_metrics[k])
        all_val_metrics_np['perm_loss'].append(np.repeat(loss_item['perm_loss'].item(), batch_cur_size))
        all_val_metrics_np['overlap_loss'].append(np.repeat(loss_item['overlap_loss'].item(), batch_cur_size))
        all_val_metrics_np['c_s_cd_loss'].append(np.repeat(loss_item['c_s_cd_loss'].item(), batch_cur_size))
        all_val_metrics_np['f_s_cd_loss'].append(np.repeat(loss_item['f_s_cd_loss'].item(), batch_cur_size))
        all_val_metrics_np['c_r_cd_loss'].append(np.repeat(loss_item['c_r_cd_loss'].item(), batch_cur_size))
        all_val_metrics_np['f_r_cd_loss'].append(np.repeat(loss_item['f_r_cd_loss'].item(), batch_cur_size))
        all_val_metrics_np['overlap_prob_loss'].append(np.repeat(loss_item['overlap_prob_loss'].item(), batch_cur_size))
        all_val_metrics_np['kl_loss'].append(np.repeat(loss_item['kl_loss'].item(), batch_cur_size))
        all_val_metrics_np['label'].append(Label)
        all_val_metrics_np['infertime'].append(np.repeat(infer_time / batch_cur_size, batch_cur_size))

    all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
    summary_metrics = summarize_metrics(all_val_metrics_np)

    eval_log = '[Metric]'
    for k in summary_metrics:
        if k.endswith('loss') or k.startswith('acc'):
            eval_log += ' Mean-' + k + ': {:.4f}'.format(summary_metrics[k])
    print(eval_log)

    print_metrics(summary_metrics)
    if metric_is_save:
        np.save(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + save_filetime + '_metric')),
                all_val_metrics_np)

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return summary_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point could registration evaluation code.')
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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)

    torch.manual_seed(cfg.RANDOM_SEED)

    pc_dataset = get_datasets(partition='test',
                              num_points=cfg.DATASET.POINT_NUM,
                              unseen=cfg.DATASET.UNSEEN,
                              noise_type=cfg.DATASET.NOISE_TYPE,
                              rot_mag=cfg.DATASET.ROT_MAG,
                              trans_mag=cfg.DATASET.TRANS_MAG,
                              partial_p_keep=cfg.DATASET.PARTIAL_P_KEEP)

    dataloader = get_dataloader(pc_dataset, phase='test')

    model = PipeLine()
    model = model.cuda()

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        metrics = eval_model(model, dataloader,
                             eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                             metric_is_save=True,
                             save_filetime=now_time)
