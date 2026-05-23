from test_joint import build_parser, main


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(report_mode="cr")
    opts = parser.parse_args()
    main(opts)
