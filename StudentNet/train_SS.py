import argparse
import os
from types import SimpleNamespace


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_sz', type=int, default=32, help='batch size used for training')

    parser.add_argument('--input_data_folder', type=str, default='/data/zzy/Datasets/M3R-CR/M3M-CR/train')
    parser.add_argument('--val_input_data_folder', type=str, default='/data/zzy/Datasets/M3R-CR/M3M-CR/test')
    parser.add_argument('--train_list_filepath', type=str, default='/data/zzy/Datasets/M3R-CR/M3M-CR/train.csv')
    parser.add_argument('--val_list_filepath', type=str, default='/data/zzy/Datasets/M3R-CR/M3M-CR/test.csv')
    parser.add_argument('--is_load_SAR', type=bool, default=True)
    parser.add_argument('--is_upsample_SAR', type=bool, default=True)
    parser.add_argument('--is_load_landcover', type=bool, default=True)
    parser.add_argument('--is_upsample_landcover', type=bool, default=False)
    parser.add_argument('--lc_level', type=str, default='1')
    parser.add_argument('--is_load_cloudmask', type=bool, default=True)
    parser.add_argument('--load_size', type=int, default=300)
    parser.add_argument('--crop_size', type=int, default=160)

    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of optimizer')
    parser.add_argument('--lr_step', type=int, default=5, help='lr decay rate')
    parser.add_argument('--lr_start_epoch_decay', type=int, default=10, help='epoch to start lr decay')
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=2)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument(
        '--teacher_pretrained_model',
        type=str,
        default='./checkpoints/TeacherNet/best_semantic_net.pth',
    )
    parser.add_argument('--save_model_dir', type=str, default='/home/zzy/zyzhang/CloudSeg/checkpoints/StudentNet')

    parser.add_argument('--continue_train_checkpoint', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--encoder_weights', type=str, default=None)
    parser.add_argument('--use_cloud_removal', type=int, choices=[0, 1], default=1)
    parser.add_argument('--use_kd', type=int, choices=[0, 1], default=1)

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    return parser


def main(opts=None):
    if opts is None:
        opts = build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    import torch
    from dataloader import get_filelist, TrainDataset, ValDataset
    from generic_train import Generic_Train
    from model_SS_net import ModelSSNet
    from model_base import print_options, seed_torch

    print_options(opts)
    seed_torch()

    train_filelist = get_filelist(opts.train_list_filepath)
    val_filelist = get_filelist(opts.val_list_filepath)

    train_data = TrainDataset(opts, train_filelist)
    val_opts = SimpleNamespace(**vars(opts))
    val_opts.input_data_folder = opts.val_input_data_folder
    val_data = ValDataset(val_opts, val_filelist)
    print("Train set: %d, Val set: %d" % (len(train_data), len(val_data)))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=True,
        num_workers=opts.num_workers,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        drop_last=True,
    )

    model = ModelSSNet(opts)
    Generic_Train(model, opts, train_dataloader, val_dataloader).train()


if __name__ == "__main__":
    main()
