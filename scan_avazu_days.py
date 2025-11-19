
import argparse
from collections import Counter
import pandas as pd

# scan_avazu_days function to scan the Avazu train.gz file and print the impression counts per day
def scan_avazu_days(file_path: str, chunksize: int = 500_000) -> None: 
    
    day_counts = Counter()

    print(f"Scanning file: {file_path}")
    print(f"Using chunksize={chunksize}")

    # Extract 'hour' column for this scan
    for i, chunk in enumerate(
        pd.read_csv(
            file_path,
            compression="gzip" if file_path.endswith(".gz") else None,
            chunksize=chunksize,
            usecols=["hour"],
        )
    ):
        # Avazu "hour" format: YYMMDDHH 
        chunk["day"] = chunk["hour"] // 100
        day_counts.update(chunk["day"].astype(str))

        if (i + 1) % 20 == 0:
            print(f"Processed {(i + 1) * chunksize:,} rows...")

    print("\nImpressions per day (most common first):")
    for day, count in day_counts.most_common():
        print(f"{day}  {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan Avazu train.gz and show impressions per day."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to Avazu train.gz or train.csv file.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Number of rows to load per chunk (default: 500000).",        )
    args = parser.parse_args()

    scan_avazu_days(args.file_path, args.chunksize)


if __name__ == "__main__":
    main()
