import argparse
from cmapps_telemetry_anomaly_detection.data_ingestion.kaggle_data_import import download_dataset
from cmapps_telemetry_anomaly_detection.data_preprocessing.data_preprocess import run_preprocessing
from cmapps_telemetry_anomaly_detection.feature_extraction.feature_extraction import run_feature_extraction


def main():
    # Create top-level parser
    p = argparse.ArgumentParser(prog="cmapps-tad")
    # Add subcommands
    sub = p.add_subparsers(dest="cmd")

    # Create subcommands ONCE
    download_parser = sub.add_parser("download", help="Download CMAPSS dataset from Kaggle")
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist (ignores .gitkeep)",
    )

    # Add other subcommands
    sub.add_parser("train", help="Train anomaly model(s)")
    sub.add_parser("score", help="Score anomalies")
    sub.add_parser("preprocess", help="Preprocess raw CMAPSS data")
    sub.add_parser("features", help="Extract features from processed data")



    args = p.parse_args()

    if args.cmd == "download":
        download_dataset(force=args.force)  # pass force flag
    elif args.cmd == "train":
        print("TODO: train anomaly model")
    elif args.cmd == "score":
        print("TODO: score anomalies")
    elif args.cmd is None:
        print("No command provided. Use --help for usage information.")
    elif args.cmd == "preprocess":
        run_preprocessing()
    elif args.cmd == "features":
        run_feature_extraction()
    else:
        p.print_help()