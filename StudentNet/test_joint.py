import argparse
import csv
import importlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageEnhance
from tqdm import tqdm

from model_SS_net import ModelSSNet


def compute_window_starts(full_size, window_size, step_size):
    starts = list(range(0, max(full_size - window_size, 0) + 1, step_size))
    last_start = full_size - window_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def rescale_start(value, input_patch_size, output_patch_size):
    scaled = value * output_patch_size / input_patch_size
    assert scaled.is_integer()
    return int(scaled)


def infer_dataset_name(input_data_folder):
    dataset_path = Path(input_data_folder).expanduser()
    split_names = {"train", "val", "valid", "validation", "test"}
    if dataset_path.name.lower() in split_names and dataset_path.parent.name:
        return dataset_path.parent.name
    return dataset_path.name or "dataset"


def resolve_export_dir(opts):
    export_dir = getattr(opts, "export_dir", None)
    if export_dir:
        return Path(export_dir).expanduser().resolve()

    dataset_name = getattr(opts, "dataset_name", None) or infer_dataset_name(opts.input_data_folder)
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "results" / dataset_name


def get_segmentation_palette(opts, num_classes):
    default_ignore = np.array([0, 0, 0], dtype=np.uint8)

    if getattr(opts, "lc_level", "") == "2":
        palette = np.array(
            [
                [0, 100, 0],
                [255, 187, 34],
                [255, 255, 76],
                [240, 150, 255],
                [250, 0, 0],
                [180, 180, 180],
                [240, 240, 240],
                [0, 100, 200],
                [0, 150, 160],
                [0, 207, 117],
                [250, 230, 160],
            ],
            dtype=np.uint8,
        )
    elif getattr(opts, "lc_level", "") == "1":
        palette = np.array(
            [
                [0, 100, 0],
                [255, 220, 80],
                [240, 150, 255],
                [250, 0, 0],
                [180, 180, 180],
                [0, 100, 200],
            ],
            dtype=np.uint8,
        )
    elif int(getattr(opts, "num_classes", num_classes)) == 7:
        palette = np.array(
            [
                [34, 139, 34],
                [255, 187, 34],
                [255, 255, 76],
                [240, 150, 255],
                [250, 0, 0],
                [0, 100, 200],
                [180, 180, 180],
            ],
            dtype=np.uint8,
        )
    else:
        palette = np.array(
            [
                [0, 100, 0],
                [255, 187, 34],
                [255, 255, 76],
                [240, 150, 255],
                [250, 0, 0],
                [0, 100, 200],
            ],
            dtype=np.uint8,
        )

    if palette.shape[0] < num_classes:
        extra = np.zeros((num_classes - palette.shape[0], 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    return palette, default_ignore


def optical_to_uint8_rgb(optical_data, brightness=3.0, rgb_bands=(2, 1, 0)):
    if torch.is_tensor(optical_data):
        optical_data = optical_data.detach().cpu().float().numpy()
    rgb = np.asarray(optical_data, dtype=np.float32)[list(rgb_bands)]
    rgb = np.clip(rgb * float(brightness), 0.0, 1.0)
    rgb = np.moveaxis(rgb, 0, -1)
    return np.ascontiguousarray((rgb * 255.0).round().astype(np.uint8))


def single_channel_to_uint8_gray(single_channel_data):
    if torch.is_tensor(single_channel_data):
        single_channel_data = single_channel_data.detach().cpu().float().numpy()
    gray = np.asarray(single_channel_data, dtype=np.float32)
    gray = np.clip(gray, 0.0, 1.0)
    return np.ascontiguousarray((gray * 255.0).round().astype(np.uint8))


def labels_to_rgb_image(label_data, palette, ignore_color):
    if torch.is_tensor(label_data):
        label_data = label_data.detach().cpu().long().numpy()
    labels = np.asarray(label_data, dtype=np.int64)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    ignore_mask = labels == 255
    valid_mask = (~ignore_mask) & (labels >= 0) & (labels < palette.shape[0])
    rgb[valid_mask] = palette[labels[valid_mask]]
    rgb[ignore_mask] = ignore_color
    return np.ascontiguousarray(rgb)


def enhance_rgb_contrast(rgb_uint8, contrast=1.0):
    image = Image.fromarray(rgb_uint8)
    if contrast is not None and float(contrast) != 1.0:
        image = ImageEnhance.Contrast(image).enhance(float(contrast))
    return np.array(image)


class ResultExporter:
    def __init__(self, opts, num_classes):
        self.opts = opts
        self.num_classes = num_classes
        self.export_payload = getattr(opts, "export_payload", "pred_only")
        self.optical_brightness = float(getattr(opts, "optical_brightness", 3.0))
        self.export_root = resolve_export_dir(opts)
        self.palette, self.ignore_color = get_segmentation_palette(opts, num_classes)
        self.name_counts = defaultdict(int)
        self.subdirs = {
            "cloudfree_pred_rgb": self.export_root / "cloudfree_pred_rgb",
            "seg_pred_rgb": self.export_root / "seg_pred_rgb",
        }
        if self.export_payload == "all":
            self.subdirs.update(
                {
                    "cloudy_rgb": self.export_root / "cloudy_rgb",
                    "sar_mean": self.export_root / "sar_mean",
                    "cloudfree_gt_rgb": self.export_root / "cloudfree_gt_rgb",
                    "seg_gt_rgb": self.export_root / "seg_gt_rgb",
                    "cloudmask": self.export_root / "cloudmask",
                }
            )
        for path in self.subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.export_root / "manifest.csv"
        self.config_path = self.export_root / "export_config.json"
        with self.manifest_path.open("w", newline="") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=[
                    "sample_name",
                    "file_name",
                    "cloudfree_path",
                    "cloudy_path",
                    "sar_path",
                    "landcover_path",
                    "cloudmask_path",
                ],
            )
            writer.writeheader()
        self.config_path.write_text(
            json.dumps(
                {
                    "dataset_name": getattr(self.opts, "dataset_name", None)
                    or infer_dataset_name(self.opts.input_data_folder),
                    "input_data_folder": str(Path(self.opts.input_data_folder).expanduser().resolve()),
                    "dataloader_module": getattr(self.opts, "dataloader_module", "dataloader"),
                    "export_payload": self.export_payload,
                    "optical_brightness": self.optical_brightness,
                    "is_load_SAR": bool(getattr(self.opts, "is_load_SAR", True)),
                    "is_upsample_SAR": bool(getattr(self.opts, "is_upsample_SAR", True)),
                    "is_load_landcover": bool(getattr(self.opts, "is_load_landcover", True)),
                    "is_upsample_landcover": bool(getattr(self.opts, "is_upsample_landcover", False)),
                    "is_load_cloudmask": bool(getattr(self.opts, "is_load_cloudmask", True)),
                    "load_size": int(getattr(self.opts, "load_size", 300)),
                    "crop_size": int(getattr(self.opts, "crop_size", 300)),
                    "lc_level": getattr(self.opts, "lc_level", None),
                    "num_classes": int(self.num_classes),
                },
                indent=2,
            )
            + "\n"
        )

    def _unique_stem(self, file_name):
        stem = Path(file_name).stem
        count = self.name_counts[stem]
        self.name_counts[stem] += 1
        if count == 0:
            return stem
        return f"{stem}__{count}"

    def _to_uint8_rgb(self, optical_tensor):
        return optical_to_uint8_rgb(optical_tensor, brightness=self.optical_brightness)

    @staticmethod
    def _to_uint8_gray(single_channel_tensor):
        return single_channel_to_uint8_gray(single_channel_tensor)

    def _labels_to_rgb(self, label_tensor):
        return labels_to_rgb_image(label_tensor, self.palette, self.ignore_color)

    @staticmethod
    def _save_image(path, array):
        Image.fromarray(array).save(path)

    @staticmethod
    def _get_source_value(source_paths, key, batch_idx):
        if source_paths is None or key not in source_paths:
            return ""
        value = source_paths[key]
        if isinstance(value, (list, tuple)):
            return str(value[batch_idx])
        return str(value)

    def _append_manifest_row(self, sample_name, file_name, source_paths, batch_idx):
        row = {
            "sample_name": sample_name,
            "file_name": file_name,
            "cloudfree_path": self._get_source_value(source_paths, "cloudfree_path", batch_idx),
            "cloudy_path": self._get_source_value(source_paths, "cloudy_path", batch_idx),
            "sar_path": self._get_source_value(source_paths, "sar_path", batch_idx),
            "landcover_path": self._get_source_value(source_paths, "landcover_path", batch_idx),
            "cloudmask_path": self._get_source_value(source_paths, "cloudmask_path", batch_idx),
        }
        with self.manifest_path.open("a", newline="") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=list(row.keys()))
            writer.writerow(row)

    def export_batch(
        self,
        file_names,
        cloudy_data,
        sar_data,
        cloudfree_gt,
        cloudfree_pred,
        seg_gt,
        seg_pred,
        cloudmask_data=None,
        source_paths=None,
    ):
        for batch_idx, file_name in enumerate(file_names):
            stem = self._unique_stem(file_name)
            self._save_image(
                self.subdirs["cloudfree_pred_rgb"] / f"{stem}.png",
                self._to_uint8_rgb(cloudfree_pred[batch_idx]),
            )
            self._save_image(
                self.subdirs["seg_pred_rgb"] / f"{stem}.png",
                self._labels_to_rgb(seg_pred[batch_idx]),
            )
            self._append_manifest_row(stem, file_name, source_paths, batch_idx)

            if self.export_payload == "all":
                self._save_image(
                    self.subdirs["cloudy_rgb"] / f"{stem}.png",
                    self._to_uint8_rgb(cloudy_data[batch_idx]),
                )
                self._save_image(
                    self.subdirs["cloudfree_gt_rgb"] / f"{stem}.png",
                    self._to_uint8_rgb(cloudfree_gt[batch_idx]),
                )
                sar_mean = sar_data[batch_idx].mean(dim=0)
                self._save_image(
                    self.subdirs["sar_mean"] / f"{stem}.png",
                    self._to_uint8_gray(sar_mean),
                )
                self._save_image(
                    self.subdirs["seg_gt_rgb"] / f"{stem}.png",
                    self._labels_to_rgb(seg_gt[batch_idx]),
                )

                if cloudmask_data is not None:
                    self._save_image(
                        self.subdirs["cloudmask"] / f"{stem}.png",
                        self._to_uint8_gray(cloudmask_data[batch_idx]),
                    )


class RunningCloudRemovalMetrics:
    def __init__(self, channels, data_range=1.0, window_size=11, sigma=1.5, device="cuda"):
        self.channels = channels
        self.data_range = data_range
        self.device = device
        self.kernel = self._build_kernel(window_size, sigma).to(device=device)
        self.reset()

    def _build_kernel(self, window_size, sigma):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        kernel_2d = torch.outer(gaussian_1d, gaussian_1d)
        return kernel_2d.expand(self.channels, 1, window_size, window_size).contiguous()

    def reset(self):
        self.sum_psnr = torch.zeros((), dtype=torch.float64, device=self.device)
        self.sum_ssim = torch.zeros((), dtype=torch.float64, device=self.device)
        self.sum_sam = torch.zeros((), dtype=torch.float64, device=self.device)
        self.sum_mae = torch.zeros((), dtype=torch.float64, device=self.device)
        self.count = torch.zeros((), dtype=torch.float64, device=self.device)

    def _ssim_per_image(self, pred, target):
        padding = self.kernel.shape[-1] // 2
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        mu_pred = F.conv2d(pred, self.kernel, padding=padding, groups=self.channels)
        mu_target = F.conv2d(target, self.kernel, padding=padding, groups=self.channels)

        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, self.kernel, padding=padding, groups=self.channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, self.kernel, padding=padding, groups=self.channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.kernel, padding=padding, groups=self.channels) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        )
        return ssim_map.mean(dim=(1, 2, 3))

    def update(self, pred, target):
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

        mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
        mae = (pred - target).abs().mean(dim=(1, 2, 3))
        psnr = 10.0 * torch.log10((self.data_range ** 2) / mse.clamp_min(1e-12))

        dot = (pred * target).sum(dim=1)
        pred_norm = pred.pow(2).sum(dim=1).sqrt()
        target_norm = target.pow(2).sum(dim=1).sqrt()
        cosine = dot / (pred_norm * target_norm).clamp_min(1e-8)
        cosine = cosine.clamp(-1.0, 1.0)
        sam = torch.rad2deg(torch.arccos(cosine)).mean(dim=(1, 2))

        ssim = self._ssim_per_image(pred, target)

        batch_size = pred.shape[0]
        self.sum_psnr += psnr.sum(dtype=torch.float64)
        self.sum_ssim += ssim.sum(dtype=torch.float64)
        self.sum_sam += sam.sum(dtype=torch.float64)
        self.sum_mae += mae.sum(dtype=torch.float64)
        self.count += batch_size

    def averages(self):
        denom = self.count.clamp_min(1.0)
        return {
            "PSNR": self.sum_psnr / denom,
            "SSIM": self.sum_ssim / denom,
            "SAM": self.sum_sam / denom,
            "MAE": self.sum_mae / denom,
        }


class RunningSegMetricsGPU:
    def __init__(self, n_classes, device="cuda"):
        self.n_classes = n_classes
        self.device = device
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.float64, device=self.device)
        self.confusion_matrix_cloudfree = torch.zeros_like(self.confusion_matrix)
        self.confusion_matrix_cloudy = torch.zeros_like(self.confusion_matrix)

    def _update_hist(self, hist, label_true, label_pred, region_mask=None):
        valid = (label_true >= 0) & (label_true < self.n_classes)
        if region_mask is not None:
            valid = valid & region_mask
        if valid.any():
            bins = torch.bincount(
                (self.n_classes * label_true[valid] + label_pred[valid]).view(-1),
                minlength=self.n_classes ** 2,
            ).reshape(self.n_classes, self.n_classes)
            hist += bins.to(dtype=torch.float64)

    def update(self, label_true, label_pred, cloudmask=None):
        self._update_hist(self.confusion_matrix, label_true, label_pred)
        if cloudmask is not None:
            self._update_hist(self.confusion_matrix_cloudfree, label_true, label_pred, cloudmask == 0)
            self._update_hist(self.confusion_matrix_cloudy, label_true, label_pred, cloudmask == 1)

    def get_results(self, hist):
        hist = hist.to(dtype=torch.float64)
        total = hist.sum().clamp_min(1.0)
        acc = torch.diag(hist).sum() / total
        acc_cls = torch.diag(hist) / hist.sum(dim=1).clamp_min(1.0)
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).clamp_min(1.0)
        freq = hist.sum(dim=1) / total
        fwavacc = (freq * iu).sum()
        return {
            "Overall Acc": acc,
            "Mean Acc": acc_cls.mean(),
            "FreqW Acc": fwavacc,
            "Mean IoU": iu.mean(),
            "Class Acc": acc_cls,
            "Class IoU": iu,
        }


class JointEvaluator:
    def __init__(self, model, opts, dataloader):
        self.model = model
        self.opts = opts
        self.dataloader = dataloader
        self.device = self.model.cloudy_data.device if hasattr(self.model, "cloudy_data") else "cuda"

        self.seg_metric = RunningSegMetricsGPU(self.model.num_classes, device="cuda")
        self.cr_metric = RunningCloudRemovalMetrics(
            channels=int(getattr(self.opts, "optical_channels", 4)),
            device="cuda",
        )

        self.opts.output_patch_size = self.opts.model_train_size
        if not self.opts.is_upsample_landcover:
            self.opts.output_patch_size = self.opts.model_train_size * 3 / 10
            assert self.opts.output_patch_size.is_integer()
            self.opts.output_patch_size = int(self.opts.output_patch_size)

        self.opts.output_size = self.opts.load_size
        if not self.opts.is_upsample_landcover:
            self.opts.output_size = self.opts.load_size * 3 / 10
            assert self.opts.output_size.is_integer()
            self.opts.output_size = int(self.opts.output_size)

        self.window_size = self.opts.model_train_size
        self.step_size = self.window_size // 2
        self.y_starts = compute_window_starts(self.opts.crop_size, self.window_size, self.step_size)
        self.x_starts = compute_window_starts(self.opts.crop_size, self.window_size, self.step_size)
        self.result_exporter = ResultExporter(opts, self.model.num_classes) if opts.export_results else None

    def _prepare_segmentation_tensors(self, pred_seg):
        seg_gt = self.model.landcover_data
        cloudmask = self.model.cloudmask_data

        if getattr(self.opts, "seg_eval_resolution", "label") == "3m":
            target_h, target_w = self.model.cloudy_data.shape[-2:]
            pred_seg = F.interpolate(
                pred_seg,
                size=[target_h, target_w],
                mode="bilinear",
                align_corners=False,
            )
            if seg_gt.shape[-2:] != (target_h, target_w):
                seg_gt = F.interpolate(
                    seg_gt.unsqueeze(1).float(),
                    size=[target_h, target_w],
                    mode="nearest",
                ).squeeze(1).long()
            if cloudmask is not None and cloudmask.shape[-2:] != (target_h, target_w):
                cloudmask = F.interpolate(
                    cloudmask.unsqueeze(1).float(),
                    size=[target_h, target_w],
                    mode="nearest",
                ).squeeze(1)
            return pred_seg, seg_gt, cloudmask

        if not self.opts.is_upsample_landcover and cloudmask is not None:
            cloudmask = F.interpolate(
                cloudmask.unsqueeze(1),
                size=[self.opts.output_size, self.opts.output_size],
                mode="nearest",
            ).squeeze(1)
        return pred_seg, seg_gt, cloudmask

    @torch.no_grad()
    def predict_joint(self, optical_data, sar_data):
        if self.opts.crop_size == self.opts.model_train_size:
            pred_seg, pred_cr, _ = self.model.forward(
                optical_data=optical_data,
                SAR_data=sar_data,
                output_shape=[self.opts.output_patch_size, self.opts.output_patch_size],
                return_all=True,
            )
            return pred_seg, pred_cr

        batch_size = optical_data.shape[0]
        pred_seg_sum = torch.zeros(
            batch_size,
            self.model.num_classes,
            self.opts.output_size,
            self.opts.output_size,
            device=optical_data.device,
        )
        pred_seg_count = torch.zeros(batch_size, 1, self.opts.output_size, self.opts.output_size, device=optical_data.device)
        pred_cr_sum = torch.zeros_like(optical_data)
        pred_cr_count = torch.zeros(batch_size, 1, optical_data.shape[2], optical_data.shape[3], device=optical_data.device)

        for y_start in self.y_starts:
            for x_start in self.x_starts:
                optical_patch = optical_data[..., y_start : y_start + self.window_size, x_start : x_start + self.window_size]
                sar_patch = sar_data[..., y_start : y_start + self.window_size, x_start : x_start + self.window_size]
                pred_seg_patch, pred_cr_patch, _ = self.model.forward(
                    optical_data=optical_patch,
                    SAR_data=sar_patch,
                    output_shape=[self.opts.output_patch_size, self.opts.output_patch_size],
                    return_all=True,
                )

                seg_y_start = rescale_start(y_start, self.opts.model_train_size, self.opts.output_patch_size)
                seg_x_start = rescale_start(x_start, self.opts.model_train_size, self.opts.output_patch_size)

                pred_seg_sum[
                    ...,
                    seg_y_start : seg_y_start + self.opts.output_patch_size,
                    seg_x_start : seg_x_start + self.opts.output_patch_size,
                ] += pred_seg_patch
                pred_seg_count[
                    ...,
                    seg_y_start : seg_y_start + self.opts.output_patch_size,
                    seg_x_start : seg_x_start + self.opts.output_patch_size,
                ] += 1

                pred_cr_sum[..., y_start : y_start + self.window_size, x_start : x_start + self.window_size] += pred_cr_patch
                pred_cr_count[..., y_start : y_start + self.window_size, x_start : x_start + self.window_size] += 1

        pred_seg = pred_seg_sum / pred_seg_count.clamp_min(1)
        pred_cr = pred_cr_sum / pred_cr_count.clamp_min(1)
        return pred_seg, pred_cr

    @torch.no_grad()
    def evaluate(self):
        self.model.net_G.eval()
        torch.backends.cudnn.benchmark = True
        progress = tqdm(self.dataloader, total=len(self.dataloader), dynamic_ncols=True)

        for _, parameter in self.model.net_G.named_parameters():
            parameter.requires_grad = False

        for batch in progress:
            self.model.set_input(batch)
            pred_seg, pred_cr = self.predict_joint(self.model.cloudy_data, self.model.SAR_data)
            pred_seg_metric, seg_gt_metric, cloudmask_metric = self._prepare_segmentation_tensors(pred_seg)
            pred_label = pred_seg_metric.argmax(dim=1)
            self.seg_metric.update(seg_gt_metric, pred_label, cloudmask_metric)
            self.cr_metric.update(pred_cr, self.model.cloudfree_data)

            if self.result_exporter is not None:
                seg_gt_export = seg_gt_metric
                seg_pred_export = pred_label
                cloudmask_export = cloudmask_metric

                if seg_gt_export.shape[-2:] != (self.opts.load_size, self.opts.load_size):
                    seg_gt_export = F.interpolate(
                        seg_gt_export.unsqueeze(1).float(),
                        size=[self.opts.load_size, self.opts.load_size],
                        mode="nearest",
                    ).squeeze(1).long()
                    seg_pred_export = F.interpolate(
                        seg_pred_export.unsqueeze(1).float(),
                        size=[self.opts.load_size, self.opts.load_size],
                        mode="nearest",
                    ).squeeze(1).long()
                if cloudmask_export is not None and cloudmask_export.shape[-2:] != (self.opts.load_size, self.opts.load_size):
                    cloudmask_export = F.interpolate(
                        cloudmask_export.unsqueeze(1).float(),
                        size=[self.opts.load_size, self.opts.load_size],
                        mode="nearest",
                    ).squeeze(1)

                self.result_exporter.export_batch(
                    file_names=self.model.file_name,
                    cloudy_data=self.model.cloudy_data,
                    sar_data=self.model.SAR_data,
                    cloudfree_gt=self.model.cloudfree_data,
                    cloudfree_pred=pred_cr,
                    seg_gt=seg_gt_export,
                    seg_pred=seg_pred_export,
                    cloudmask_data=cloudmask_export,
                    source_paths=getattr(self.model, "source_paths", None),
                )

            seg_all = self.seg_metric.get_results(self.seg_metric.confusion_matrix)
            seg_cf = self.seg_metric.get_results(self.seg_metric.confusion_matrix_cloudfree)
            seg_cy = self.seg_metric.get_results(self.seg_metric.confusion_matrix_cloudy)
            cr = self.cr_metric.averages()
            progress.set_postfix(
                seg_miou=f"{seg_all['Mean IoU'].item():.4f}",
                cf_miou=f"{seg_cf['Mean IoU'].item():.4f}",
                cy_miou=f"{seg_cy['Mean IoU'].item():.4f}",
                psnr=f"{cr['PSNR'].item():.2f}",
                ssim=f"{cr['SSIM'].item():.4f}",
                sam=f"{cr['SAM'].item():.2f}",
                mae=f"{cr['MAE'].item():.4f}",
            )

        seg_all = self.seg_metric.get_results(self.seg_metric.confusion_matrix)
        seg_cf = self.seg_metric.get_results(self.seg_metric.confusion_matrix_cloudfree)
        seg_cy = self.seg_metric.get_results(self.seg_metric.confusion_matrix_cloudy)
        cr = self.cr_metric.averages()
        return seg_all, seg_cf, seg_cy, cr


def print_seg_results(name, results):
    print(name)
    print(
        "Overall Acc: %.6f\tMean Acc: %.6f\tFreqW Acc: %.6f\tMean IoU: %.6f"
        % (
            results["Overall Acc"].item(),
            results["Mean Acc"].item(),
            results["FreqW Acc"].item(),
            results["Mean IoU"].item(),
        )
    )
    print("Class Acc:", results["Class Acc"].detach().cpu().numpy())
    print("Class IoU:", results["Class IoU"].detach().cpu().numpy())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, default="/data/zzy/Datasets/M3R-CR/M3M-CR/test")
    parser.add_argument("--is_load_SAR", type=bool, default=True)
    parser.add_argument("--is_upsample_SAR", type=bool, default=True)
    parser.add_argument("--is_load_landcover", type=bool, default=True)
    parser.add_argument("--is_upsample_landcover", type=bool, default=False)
    parser.add_argument("--lc_level", type=str, default="1")
    parser.add_argument("--is_load_cloudmask", type=bool, default=True)
    parser.add_argument("--load_size", type=int, default=300)
    parser.add_argument("--crop_size", type=int, default=300)
    parser.add_argument("--model_train_size", type=int, default=160)
    parser.add_argument("--test_list_filepath", type=str, default="/data/zzy/Datasets/M3R-CR/M3M-CR/test.csv")
    parser.add_argument("--pretrained_model", type=str, default="/home/zzy/zyzhang/CloudSeg/pretrained/StudentNet.pth")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--report_mode", choices=["joint", "seg", "cr"], default="joint")
    parser.add_argument("--dataloader_module", type=str, default="dataloader")
    parser.add_argument("--seg_eval_resolution", choices=["label", "3m"], default="label")
    parser.add_argument("--export_results", action="store_true")
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--export_payload", choices=["pred_only", "all"], default="pred_only")
    parser.add_argument("--optical_brightness", type=float, default=3.0)
    return parser


def finalize_legacy_opts(opts):
    # The joint evaluator always computes both tasks to avoid duplicated forward passes.
    opts.is_load_landcover = True
    opts.is_load_cloudmask = True
    return opts


def main(opts=None):
    if opts is None:
        opts = build_parser().parse_args()
    opts = finalize_legacy_opts(opts)

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    torch.cuda.set_device(int(str(opts.gpu_ids).split(",")[0]))

    dataloader_module = importlib.import_module(opts.dataloader_module)
    test_filelist = dataloader_module.get_filelist(opts.test_list_filepath)
    test_data = dataloader_module.ValDataset(opts, test_filelist)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=opts.num_workers > 0,
    )
    print(f"Test set: {len(test_data)}")

    model = ModelSSNet(opts)
    evaluator = JointEvaluator(model, opts, test_dataloader)
    if opts.export_results:
        print(f"Export directory: {evaluator.result_exporter.export_root}")
    seg_all, seg_cf, seg_cy, cr = evaluator.evaluate()

    if opts.report_mode in {"joint", "seg"}:
        print_seg_results("All Regions", seg_all)
        print_seg_results("Cloud-Free Regions", seg_cf)
        print_seg_results("Cloudy Regions", seg_cy)
    if opts.report_mode in {"joint", "cr"}:
        print(
            "Cloud Removal\nPSNR: %.4f dB\tSSIM: %.6f\tSAM: %.4f deg\tMAE: %.6f"
            % (cr["PSNR"].item(), cr["SSIM"].item(), cr["SAM"].item(), cr["MAE"].item())
        )


if __name__ == "__main__":
    main()
