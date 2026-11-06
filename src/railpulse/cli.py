"""Use railpulse fetch, then railpulse train."""
import argparse
import json
from .io import download_sncf
from .prepare import load_and_validate
from .model import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="RailPulse public railway forecast")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--output", default="data/regularite-mensuelle-intercites.csv")
    t = sub.add_parser("train")
    t.add_argument("--csv", default="data/regularite-mensuelle-intercites.csv")
    t.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    if args.command == "fetch":
        print(download_sncf(args.output))
    else:
        print(json.dumps(train_and_evaluate(load_and_validate(args.csv), args.output), indent=2))

if __name__ == "__main__":
    main()
