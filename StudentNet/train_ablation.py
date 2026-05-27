from train_SS import build_parser, main


VARIANT_DEFAULTS = {
    "cloudseg_star": {
        "use_cloud_removal": 0,
        "use_kd": 0,
        "save_model_dir": "/home/zzy/zyzhang/CloudSeg/checkpoints/StudentNet_ablation_cloudseg_star",
    },
    "cr_only": {
        "use_cloud_removal": 1,
        "use_kd": 0,
        "save_model_dir": "/home/zzy/zyzhang/CloudSeg/checkpoints/StudentNet_ablation_cr_only",
    },
    "no_kd": {
        "use_cloud_removal": 1,
        "use_kd": 0,
        "save_model_dir": "/home/zzy/zyzhang/CloudSeg/checkpoints/StudentNet_ablation_no_kd",
    },
    "kd_only": {
        "use_cloud_removal": 0,
        "use_kd": 1,
        "save_model_dir": "/home/zzy/zyzhang/CloudSeg/checkpoints/StudentNet_ablation_kd_only",
    },
    "cloudseg": {
        "use_cloud_removal": 1,
        "use_kd": 1,
        "save_model_dir": "/home/zzy/zyzhang/CloudSeg/checkpoints/StudentNet_ablation_cloudseg",
    },
}


if __name__ == "__main__":
    parser = build_parser()
    default_save_model_dir = parser.get_default("save_model_dir")
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_DEFAULTS.keys()),
        default="cr_only",
        help="Paper Table-3 ablation preset.",
    )
    opts = parser.parse_args()

    defaults = VARIANT_DEFAULTS[opts.variant]
    opts.use_cloud_removal = defaults["use_cloud_removal"]
    opts.use_kd = defaults["use_kd"]
    if opts.save_model_dir == default_save_model_dir:
        opts.save_model_dir = defaults["save_model_dir"]

    main(opts)
