import argparse
import importlib
import os
from pathlib import Path

from luojiaset_support import ensure_luojiaset_split_csvs, get_filelist


def _set_cpu_thread_env(num_threads):
    num_threads = str(max(1, int(num_threads)))
    os.environ["OMP_NUM_THREADS"] = num_threads
    os.environ["MKL_NUM_THREADS"] = num_threads
    os.environ["OPENBLAS_NUM_THREADS"] = num_threads
    os.environ["NUMEXPR_NUM_THREADS"] = num_threads
    os.environ["GDAL_NUM_THREADS"] = num_threads
    os.environ["OPJ_NUM_THREADS"] = num_threads


def _seed_worker_factory(base_seed, worker_threads):
    def _seed_worker(worker_id):
        import random

        import numpy as np

        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _seed_worker


class BaseLuojiaScript:
    dataset_root = "/mnt/ramdisk/LuojiaSET-OSFCR"
    split_dir = "artifacts/luojiaset_splits"
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    split_seed = 42
    optical_channels = 13
    num_classes = 7
    load_size = 256
    crop_size = 256
    model_train_size = 160
    batch_size = 16
    num_workers = 8
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True
    seed = 911
    deterministic = False
    worker_threads = 1
    is_load_cloudmask = True

    def _repo_root(self):
        return Path(__file__).resolve().parent

    def _default_split_paths(self):
        split_paths = ensure_luojiaset_split_csvs(
            dataset_root=self.dataset_root,
            output_dir=self._repo_root() / self.split_dir,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
        )
        return {name: str(path) for name, path in split_paths.items()}

    def _add_common_dataset_args(self, parser):
        default_splits = self._default_split_paths()
        parser.add_argument("--input_data_folder", type=str, default=self.dataset_root)
        parser.add_argument("--train_list_filepath", type=str, default=default_splits["train"])
        parser.add_argument("--val_list_filepath", type=str, default=default_splits["val"])
        parser.add_argument("--test_list_filepath", type=str, default=default_splits["test"])
        parser.add_argument("--is_load_SAR", type=bool, default=True)
        parser.add_argument("--is_upsample_SAR", type=bool, default=True)
        parser.add_argument("--is_load_landcover", type=bool, default=True)
        parser.add_argument("--is_upsample_landcover", type=bool, default=True)
        parser.add_argument("--lc_level", type=str, default="luojia")
        parser.add_argument("--is_load_cloudmask", type=bool, default=self.is_load_cloudmask)
        parser.add_argument("--load_size", type=int, default=self.load_size)
        parser.add_argument("--crop_size", type=int, default=self.crop_size)
        parser.add_argument("--model_train_size", type=int, default=self.model_train_size)
        parser.add_argument("--optical_channels", type=int, default=self.optical_channels)
        parser.add_argument("--num_classes", type=int, default=self.num_classes)
        parser.add_argument("--gpu_ids", type=str, default="0")
        parser.add_argument("--num_workers", type=int, default=self.num_workers)
        parser.add_argument("--prefetch_factor", type=int, default=self.prefetch_factor)
        parser.add_argument("--pin_memory", type=bool, default=self.pin_memory)
        parser.add_argument("--persistent_workers", type=bool, default=self.persistent_workers)
        parser.add_argument("--seed", type=int, default=self.seed)
        parser.add_argument("--deterministic", type=bool, default=self.deterministic)
        parser.add_argument("--worker_threads", type=int, default=self.worker_threads)
        return parser


class BaseLuojiaTrainScript(BaseLuojiaScript):
    crop_size = 160
    model_module_name = "model_SS_net"
    dataloader_module_name = "dataloader_luojiaset"
    generic_train_module_name = "generic_train"
    save_model_dir = ""
    teacher_pretrained_model = None

    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_sz", type=int, default=self.batch_size)
        self._add_common_dataset_args(parser)
        parser.add_argument("--optimizer", type=str, default="Adam")
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--lr_step", type=int, default=5)
        parser.add_argument("--lr_start_epoch_decay", type=int, default=10)
        parser.add_argument("--max_epochs", type=int, default=30)
        parser.add_argument("--save_freq", type=int, default=1)
        parser.add_argument("--val_freq", type=int, default=2)
        parser.add_argument("--log_iter", type=int, default=10)
        parser.add_argument("--save_model_dir", type=str, default=self.save_model_dir)
        parser.add_argument("--continue_train_checkpoint", type=str, default=None)
        parser.add_argument("--pretrained_model", type=str, default=None)
        parser.add_argument("--encoder_weights", type=str, default=None)
        parser.add_argument("--teacher_pretrained_model", type=str, default=self.teacher_pretrained_model)
        return parser

    def run(self, opts=None):
        if opts is None:
            opts = self.build_parser().parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
        _set_cpu_thread_env(opts.worker_threads)

        torch = importlib.import_module("torch")
        dataloader_module = importlib.import_module(self.dataloader_module_name)
        model_module = importlib.import_module(self.model_module_name)
        generic_train_module = importlib.import_module(self.generic_train_module_name)
        model_base_module = importlib.import_module("model_base")

        model_base_module.print_options(opts)
        model_base_module.seed_torch(seed=opts.seed, deterministic=opts.deterministic)
        torch.set_num_threads(max(1, int(opts.worker_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        train_filelist = get_filelist(opts.train_list_filepath)
        val_filelist = get_filelist(opts.val_list_filepath)

        train_data = dataloader_module.TrainDataset(opts, train_filelist)
        val_data = dataloader_module.ValDataset(opts, val_filelist)
        print("Train set: %d, Val set: %d" % (len(train_data), len(val_data)))

        dataloader_kwargs = {
            "num_workers": opts.num_workers,
            "drop_last": True,
            "pin_memory": opts.pin_memory,
            "worker_init_fn": _seed_worker_factory(opts.seed, opts.worker_threads),
        }
        generator = torch.Generator()
        generator.manual_seed(opts.seed)
        dataloader_kwargs["generator"] = generator
        if opts.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = opts.persistent_workers
            dataloader_kwargs["prefetch_factor"] = opts.prefetch_factor

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=opts.batch_sz,
            shuffle=True,
            **dataloader_kwargs,
        )
        val_dataloader_kwargs = dict(dataloader_kwargs)
        if opts.num_workers > 0:
            val_dataloader_kwargs["num_workers"] = max(1, min(2, opts.num_workers // 2))
        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_data,
            batch_size=opts.batch_sz,
            shuffle=False,
            **val_dataloader_kwargs,
        )

        model = model_module.ModelSSNet(opts)
        generic_train_module.Generic_Train(model, opts, train_dataloader, val_dataloader).train()


class BaseLuojiaTestScript(BaseLuojiaScript):
    report_mode = "joint"
    dataloader_module_name = "dataloader_luojiaset"

    def build_parser(self):
        parser = argparse.ArgumentParser()
        self._add_common_dataset_args(parser)
        parser.add_argument("--pretrained_model", type=str, required=True)
        parser.add_argument("--batch_size", type=int, default=self.batch_size)
        parser.add_argument("--report_mode", choices=["joint", "seg", "cr"], default=self.report_mode)
        parser.add_argument("--dataloader_module", type=str, default=self.dataloader_module_name)
        return parser

    def run(self, opts=None):
        if opts is None:
            opts = self.build_parser().parse_args()
        test_joint = importlib.import_module("test_joint")
        test_joint.main(opts)
