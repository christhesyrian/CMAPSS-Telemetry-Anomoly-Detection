import argparse

def main():
    p = argparse.ArgumentParser(prog="cmapps-tad")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("download")
    sub.add_parser("train")
    sub.add_parser("score")

    args = p.parse_args()

    if args.cmd == "download":
        print("TODO: download CMAPSS/Kaggle data")
    elif args.cmd == "train":
        print("TODO: train anomaly model")
    elif args.cmd == "score":
        print("TODO: score anomalies")
    else:
        p.print_help()