import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from rasterio.windows import Window


BUCKET_NAMES = ("0-20%", "20%-40%", "40%-60%", "60%-80%", "80%-100%")
CORE_MODALITIES = ("s1", "s2", "s2_cloudy")
OPTIONAL_MODALITIES = ("cloud_detection_results", "land_cover_maps")
DATA_MODALITIES = CORE_MODALITIES + OPTIONAL_MODALITIES
CSV_MODALITIES = ("s2", "s2_cloudy", "s1", "land_cover_maps", "cloud_detection_results")


def get_filelist(listpath):
    filelist = []
    with open(listpath, "r") as list_csv_file:
        list_reader = csv.reader(list_csv_file)
        for item in list_reader:
            if item:
                filelist.append(item)
    return filelist


def _parse_anchor_filename(path):
    parts = path.stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected LuojiaSET filename: {path.name}")
    return parts[1], parts[3]


def _build_filename(roi_id, patch_id, modality):
    if modality == "s1":
        return f"ROIs_{roi_id}_s1_{patch_id}.tif"
    if modality == "s2":
        return f"ROIs_{roi_id}_s2_{patch_id}.tif"
    if modality == "s2_cloudy":
        return f"ROIs_{roi_id}_s2_cloudy_{patch_id}.tif"
    if modality == "cloud_detection_results":
        return f"ROIs_{roi_id}_s2_cloudy_{patch_id}.tif"
    if modality == "land_cover_maps":
        return f"ROIs_{roi_id}_s2_{patch_id}.tif"
    raise ValueError(f"Unsupported modality: {modality}")


class LuojiaSETSplitIndex:
    SUPPORTED_SPLITS = ("train", "val", "test", "all")

    def __init__(
        self,
        dataset_root,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        split_seed=42,
        sample_ratio=None,
        sample_count=None,
        sample_seed=42,
        cloud_coverage_ranges=None,
        modalities=None,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.sample_ratio = sample_ratio
        self.sample_count = sample_count
        self.sample_seed = sample_seed
        self.cloud_coverage_ranges = list(cloud_coverage_ranges or BUCKET_NAMES)
        self.modalities = list(modalities or CORE_MODALITIES)

        self._validate_configuration()
        self.data_index = self._build_data_index()

    def _validate_configuration(self):
        if not self.dataset_root.exists():
            raise ValueError(f"Root directory does not exist: {self.dataset_root}")

        for coverage in self.cloud_coverage_ranges:
            if coverage not in BUCKET_NAMES:
                raise ValueError(f"Invalid cloud coverage range: {coverage}")

        for modality in self.modalities:
            if modality not in DATA_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0, "
                f"got {ratio_sum:.6f}"
            )

        if self.sample_ratio is not None and self.sample_count is not None:
            raise ValueError("Only one of sample_ratio or sample_count can be set")

        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive")

    def _build_data_index(self):
        data_index = []

        for coverage_range in self.cloud_coverage_ranges:
            coverage_dir = self.dataset_root / coverage_range
            s2_dir = coverage_dir / "s2"

            if not s2_dir.exists():
                continue

            for s2_path in sorted(s2_dir.glob("ROIs_*_s2_p*.tif")):
                roi_id, patch_id = _parse_anchor_filename(s2_path)
                sample_info = {
                    "coverage_range": coverage_range,
                    "roi_id": roi_id,
                    "patch_id": patch_id,
                    "files": {},
                }

                all_files_exist = True
                for modality in self.modalities:
                    file_path = coverage_dir / modality / _build_filename(
                        roi_id=roi_id,
                        patch_id=patch_id,
                        modality=modality,
                    )
                    if not file_path.exists():
                        all_files_exist = False
                        break
                    sample_info["files"][modality] = file_path.relative_to(self.dataset_root).as_posix()

                if all_files_exist:
                    data_index.append(sample_info)

        return data_index

    def _resolve_sample_target(self, total_count):
        if self.sample_count is not None:
            return min(self.sample_count, total_count)

        if self.sample_ratio is None:
            return None

        ratio = float(self.sample_ratio)
        if ratio > 1.0:
            ratio = ratio / 100.0
        if ratio <= 0.0:
            raise ValueError("sample_ratio must be positive")
        if ratio > 1.0:
            raise ValueError("sample_ratio must be in (0, 1] or a percentage in (0, 100]")

        sample_target = int(round(total_count * ratio))
        if total_count > 0:
            sample_target = max(1, sample_target)
        return min(sample_target, total_count)

    def _apply_uniform_sampling(self, indices):
        target_count = self._resolve_sample_target(len(indices))
        if target_count is None or target_count >= len(indices):
            return indices

        grouped_indices = defaultdict(list)
        for idx in indices:
            coverage = self.data_index[idx]["coverage_range"]
            grouped_indices[coverage].append(idx)

        rng = random.Random(self.sample_seed)
        active_coverages = []
        for coverage in self.cloud_coverage_ranges:
            group = grouped_indices.get(coverage, [])
            if group:
                rng.shuffle(group)
                grouped_indices[coverage] = group
                active_coverages.append(coverage)

        sampled_indices = []
        while len(sampled_indices) < target_count and active_coverages:
            next_coverages = []
            for coverage in active_coverages:
                group = grouped_indices[coverage]
                if group:
                    sampled_indices.append(group.pop())
                    if len(sampled_indices) >= target_count:
                        break
                if group:
                    next_coverages.append(coverage)
            active_coverages = next_coverages

        return sampled_indices

    def build_split_indices(self, split):
        if split not in self.SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        if split == "all":
            return list(range(len(self.data_index)))

        grouped_indices = defaultdict(list)
        for idx, sample_info in enumerate(self.data_index):
            grouped_indices[sample_info["coverage_range"]].append(idx)

        rng = random.Random(self.split_seed)
        split_indices = []

        for coverage in self.cloud_coverage_ranges:
            coverage_indices = list(grouped_indices.get(coverage, []))
            rng.shuffle(coverage_indices)

            total = len(coverage_indices)
            train_end = int(total * self.train_ratio)
            val_end = train_end + int(total * self.val_ratio)

            if split == "train":
                split_indices.extend(coverage_indices[:train_end])
            elif split == "val":
                split_indices.extend(coverage_indices[train_end:val_end])
            elif split == "test":
                split_indices.extend(coverage_indices[val_end:])

        return self._apply_uniform_sampling(split_indices)

    def _record_to_row(self, sample_info):
        return [sample_info["files"][modality] for modality in CSV_MODALITIES]

    def get_split_rows(self, split):
        return [self._record_to_row(self.data_index[idx]) for idx in self.build_split_indices(split)]


def write_luojiaset_split_csvs(
    dataset_root,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=42,
    sample_ratio=None,
    sample_count=None,
    sample_seed=42,
    cloud_coverage_ranges=None,
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_index = LuojiaSETSplitIndex(
        dataset_root=dataset_root,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        sample_ratio=sample_ratio,
        sample_count=sample_count,
        sample_seed=sample_seed,
        cloud_coverage_ranges=cloud_coverage_ranges,
        modalities=DATA_MODALITIES,
    )

    split_paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    split_rows = {split_name: split_index.get_split_rows(split_name) for split_name in split_paths}

    for split_name, csv_path in split_paths.items():
        with csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(split_rows[split_name])

    manifest = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "sample_ratio": sample_ratio,
        "sample_count": sample_count,
        "sample_seed": sample_seed,
        "cloud_coverage_ranges": list(cloud_coverage_ranges or BUCKET_NAMES),
        "indexed_samples": len(split_index.data_index),
    }
    with (output_dir / "split_config.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return split_paths


def ensure_luojiaset_split_csvs(
    dataset_root,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    split_seed=42,
    sample_ratio=None,
    sample_count=None,
    sample_seed=42,
    cloud_coverage_ranges=None,
):
    output_dir = Path(output_dir).expanduser().resolve()
    split_paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    manifest_path = output_dir / "split_config.json"
    expected_manifest = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "sample_ratio": sample_ratio,
        "sample_count": sample_count,
        "sample_seed": sample_seed,
        "cloud_coverage_ranges": list(cloud_coverage_ranges or BUCKET_NAMES),
    }

    if all(path.exists() for path in split_paths.values()) and manifest_path.exists():
        with manifest_path.open("r") as handle:
            current_manifest = json.load(handle)
        current_manifest = {key: current_manifest.get(key) for key in expected_manifest}
        if current_manifest == expected_manifest:
            return split_paths

    return write_luojiaset_split_csvs(
        dataset_root=dataset_root,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        sample_ratio=sample_ratio,
        sample_count=sample_count,
        sample_seed=sample_seed,
        cloud_coverage_ranges=cloud_coverage_ranges,
    )


def collect_luojiaset_records(dataset_root):
    split_index = LuojiaSETSplitIndex(dataset_root=dataset_root, modalities=DATA_MODALITIES)
    records_by_bucket = {bucket_name: [] for bucket_name in BUCKET_NAMES}
    for sample_info in split_index.data_index:
        records_by_bucket[sample_info["coverage_range"]].append(
            split_index._record_to_row(sample_info)
        )
    return records_by_bucket


def create_luojiaset_datasets(root_dir, **kwargs):
    split_index = LuojiaSETSplitIndex(
        dataset_root=root_dir,
        train_ratio=kwargs.get("train_ratio", 0.8),
        val_ratio=kwargs.get("val_ratio", 0.1),
        test_ratio=kwargs.get("test_ratio", 0.1),
        split_seed=kwargs.get("split_seed", 42),
        sample_ratio=kwargs.get("sample_ratio"),
        sample_count=kwargs.get("sample_count"),
        sample_seed=kwargs.get("sample_seed", 42),
        cloud_coverage_ranges=kwargs.get("cloud_coverage_ranges"),
        modalities=kwargs.get("modalities", DATA_MODALITIES),
    )
    return {
        "train": split_index.build_split_indices("train"),
        "val": split_index.build_split_indices("val"),
        "test": split_index.build_split_indices("test"),
    }


def _sanitize_array(image):
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0, copy=False).astype("float32", copy=False)


def _read_raster(path, band=None, window=None):
    with rasterio.open(path, "r", driver="GTiff") as src:
        if band is None:
            image = src.read(window=window)
        else:
            image = src.read(band, window=window)
    return _sanitize_array(image)


def get_opt_image(path, window=None):
    return _read_raster(path, window=window)


def get_sar_image(path, window=None):
    return _read_raster(path, window=window)


def get_landcover_image(path, window=None):
    return _read_raster(path, band=1, window=window)


def get_cloudmask_image(path, window=None):
    return _read_raster(path, band=1, window=window)


def normalize_optical(image):
    image = np.clip(image, 0, 10000)
    return image / 10000.0


def normalize_sar(image):
    clip_min = [-25.0, -32.5]
    clip_max = [0.0, 0.0]
    normalized = image.copy()
    for channel in range(normalized.shape[0]):
        normalized[channel] = np.clip(normalized[channel], clip_min[channel], clip_max[channel])
        normalized[channel] -= clip_min[channel]
        normalized[channel] /= clip_max[channel] - clip_min[channel]
    return normalized


class LuojiaBaseDataset(Dataset):
    def __init__(self, opts, filelist, is_train):
        self.input_data_folder = Path(opts.input_data_folder)
        self.is_load_SAR = opts.is_load_SAR
        self.is_load_landcover = opts.is_load_landcover
        self.is_load_cloudmask = opts.is_load_cloudmask
        self.load_size = opts.load_size
        self.crop_size = opts.crop_size
        self.relative_filelist = [tuple(sample_paths) for sample_paths in filelist]
        self.filelist = [
            tuple(str(self.input_data_folder / relative_path) for relative_path in sample_paths)
            for sample_paths in filelist
        ]
        self.n_images = len(self.filelist)
        self.is_train = is_train

    def _get_crop_params(self):
        if self.load_size - self.crop_size <= 0:
            return 0, 0, self.crop_size, self.crop_size
        if self.is_train:
            y = random.randint(0, max(0, self.load_size - self.crop_size))
            x = random.randint(0, max(0, self.load_size - self.crop_size))
        else:
            y = max(0, self.load_size - self.crop_size) // 2
            x = max(0, self.load_size - self.crop_size) // 2
        return y, x, self.crop_size, self.crop_size

    def __getitem__(self, index):
        cloudfree_path, cloudy_path, sar_path, landcover_path, cloudmask_path = self.filelist[index]
        y, x, crop_h, crop_w = self._get_crop_params()
        window = None
        if self.load_size - self.crop_size > 0:
            window = Window(x, y, crop_w, crop_h)

        cloudfree_data = normalize_optical(get_opt_image(cloudfree_path, window=window))
        cloudy_data = normalize_optical(get_opt_image(cloudy_path, window=window))
        sar_data = normalize_sar(get_sar_image(sar_path, window=window)) if self.is_load_SAR else None
        landcover_data = get_landcover_image(landcover_path, window=window) if self.is_load_landcover else None
        cloudmask_data = get_cloudmask_image(cloudmask_path, window=window) if self.is_load_cloudmask else None

        cloudfree_data = np.ascontiguousarray(cloudfree_data)
        cloudy_data = np.ascontiguousarray(cloudy_data)
        if sar_data is not None:
            sar_data = np.ascontiguousarray(sar_data)
        if landcover_data is not None:
            landcover_data = np.ascontiguousarray(landcover_data)
        if cloudmask_data is not None:
            cloudmask_data = np.ascontiguousarray(cloudmask_data)

        results = {
            "cloudy_data": torch.from_numpy(cloudy_data),
            "cloudfree_data": torch.from_numpy(cloudfree_data),
            "file_name": Path(cloudfree_path).name,
            "source_paths": {
                "cloudfree_path": self.relative_filelist[index][0],
                "cloudy_path": self.relative_filelist[index][1],
                "sar_path": self.relative_filelist[index][2],
                "landcover_path": self.relative_filelist[index][3],
                "cloudmask_path": self.relative_filelist[index][4],
            },
        }
        if sar_data is not None:
            results["SAR_data"] = torch.from_numpy(sar_data)
        if landcover_data is not None:
            results["landcover_data"] = torch.from_numpy(landcover_data)
        if cloudmask_data is not None:
            results["cloudmask_data"] = torch.from_numpy(cloudmask_data)
        return results

    def __len__(self):
        return self.n_images


class TrainDataset(LuojiaBaseDataset):
    def __init__(self, opts, filelist):
        super().__init__(opts, filelist, is_train=True)


class ValDataset(LuojiaBaseDataset):
    def __init__(self, opts, filelist):
        super().__init__(opts, filelist, is_train=False)
