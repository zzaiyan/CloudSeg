import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import ValDataset, get_filelist
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
        self.cr_metric = RunningCloudRemovalMetrics(channels=4, device="cuda")

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

            cloudmask = self.model.cloudmask_data
            if not self.opts.is_upsample_landcover:
                cloudmask = F.interpolate(
                    cloudmask.unsqueeze(1),
                    size=[self.opts.output_size, self.opts.output_size],
                    mode="nearest",
                ).squeeze(1)

            pred_label = pred_seg.argmax(dim=1)
            self.seg_metric.update(self.model.landcover_data, pred_label, cloudmask)
            self.cr_metric.update(pred_cr, self.model.cloudfree_data)

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
    parser.add_argument("--input_data_folder", type=str, default="/remote-home/xufang/data/RS/M3M-CR/test")
    parser.add_argument("--is_load_SAR", type=bool, default=True)
    parser.add_argument("--is_upsample_SAR", type=bool, default=True)
    parser.add_argument("--is_load_landcover", type=bool, default=True)
    parser.add_argument("--is_upsample_landcover", type=bool, default=False)
    parser.add_argument("--lc_level", type=str, default="1")
    parser.add_argument("--is_load_cloudmask", type=bool, default=True)
    parser.add_argument("--load_size", type=int, default=300)
    parser.add_argument("--crop_size", type=int, default=300)
    parser.add_argument("--model_train_size", type=int, default=160)
    parser.add_argument("--test_list_filepath", type=str, default="../M3R-CR/csv/test.csv")
    parser.add_argument("--pretrained_model", type=str, default="../checkpoints/StudentNet.pth")
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--report_mode", choices=["joint", "seg", "cr"], default="joint")
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

    test_filelist = get_filelist(opts.test_list_filepath)
    test_data = ValDataset(opts, test_filelist)
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
