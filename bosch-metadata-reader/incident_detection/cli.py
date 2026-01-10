import sys
from data import load_data, scale_per_location
from models import train_by_location, load_latest_models
from visualize import detect_and_visualize

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py [train|existing] [csv_path]")
        sys.exit(1)

    mode = sys.argv[1]
    csv = sys.argv[2] if len(sys.argv) > 2 else None

    df = load_data(csv)
    df, bounds = scale_per_location(df)

    if mode == "train":
        train_by_location(df)
    else:
        models = load_latest_models()
        detect_and_visualize(df, models, bounds)

if __name__ == "__main__":
    main()
