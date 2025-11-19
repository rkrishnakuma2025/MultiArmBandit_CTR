
import argparse
from collections import Counter
from typing import List
import pandas as pd


def find_top_arms(
    file_path: str,
    target_day: int,
    arm_column: str = "site_id",
    chunksize: int = 500_000,
) -> List[str]:
    # find_top_arms function to find the most frequent arms (e.g., site_id) on a given day
    arm_counts = Counter()

    print(f"Scanning {file_path} to find top arms for day {target_day}...")
    compression = "gzip" if file_path.endswith(".gz") else None

    for i, chunk in enumerate(
        pd.read_csv(
            file_path,
            compression=compression,
            chunksize=chunksize,
            usecols=["hour", arm_column],
        )
    ):
        chunk["day"] = chunk["hour"] // 100
        day_chunk = chunk[chunk["day"] == target_day]
        if not day_chunk.empty:
            arm_counts.update(day_chunk[arm_column].astype(str))

        if (i + 1) % 20 == 0:
            print(f"  Processed {(i + 1) * chunksize:,} rows...")

    if not arm_counts:
        raise ValueError(f"No rows found for day {target_day} in {file_path}.")

    print("\nTop arms for day", target_day)
    for arm, count in arm_counts.most_common(20):
        print(f"  {arm}: {count:,} impressions")

    return [arm for arm, _ in arm_counts.most_common()]


def build_subset(
    file_path: str,
    output_path: str,
    target_day: int,
    top_arms: List[str],
    arm_column: str = "site_id",
    max_rows: int = 200_000,
    chunksize: int = 500_000,
) -> None:
    # build_subset function to build a small bandit subset for a single day and a set of top arms
    print(
        f"\nBuilding subset for day={target_day}, "
        f"{len(top_arms)} candidate arms (will stop at {max_rows} rows)..."
    )

    compression = "gzip" if file_path.endswith(".gz") else None
    rows = []
    total_kept = 0

    usecols = [
        "click",
        "hour",
        arm_column,
        "banner_pos",
        "device_type",
        "C1",
        "C15",
        "C16",
        "C17",
    ]

    for i, chunk in enumerate(
        pd.read_csv(
            file_path,
            compression=compression,
            chunksize=chunksize,
            usecols=usecols,
        )
    ):
        chunk["day"] = chunk["hour"] // 100
        # filter to target day and top arms
        mask = (chunk["day"] == target_day) & (
            chunk[arm_column].astype(str).isin(top_arms)
        )
        sub = chunk.loc[mask, :].copy()

        if not sub.empty:
            # rename arm column to a generic 'arm_id' for bandit code
            sub["arm_id"] = sub[arm_column].astype(str)
            sub = sub.drop(columns=["day"])
            rows.append(sub)
            total_kept += len(sub)

            print(f"  Collected {total_kept:,} rows so far...")

            if total_kept >= max_rows:
                break

    if not rows:
        raise ValueError(
            f"No rows collected for day {target_day} and specified arms."
        )

    df = pd.concat(rows, ignore_index=True)
    # sort by time to preserve chronological order
    df = df.sort_values("hour").reset_index(drop=True)

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    df.to_csv(output_path, index=False)
    print(f"\nSaved subset with {len(df):,} rows to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a small Avazu bandit subset for a single day and top arms "
            "(e.g., top site_id values)."
        )
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to Avazu train.gz or train.csv file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="avazu_bandit_subset.csv",
        help="Where to save the subset CSV (default: avazu_bandit_subset.csv).",
    )
    parser.add_argument(
        "--target_day",
        type=int,
        default=141022,
        help=(
            "Day code in YYMMDD format (e.g., 141022) extracted from Avazu 'hour' "
            "column by integer division by 100. Default: 141022."
        ),
    )
    parser.add_argument(
        "--arm_column",
        type=str,
        default="site_id",
        help=(
            "Column to treat as bandit arm ID (default: 'site_id'). "
            "You could also try 'banner_pos' or others."
        ),
    )
    parser.add_argument(
        "--num_arms",
        type=int,
        default=10,
        help="Number of most frequent arms to keep (default: 10).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200_000,
        help="Maximum number of rows in the subset (default: 200000).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk when reading the big file (default: 500000).",
    )
    args = parser.parse_args()

    # 1) Find top arms for the given day
    all_arms_sorted = find_top_arms(
        file_path=args.file_path,
        target_day=args.target_day,
        arm_column=args.arm_column,
        chunksize=args.chunksize,
    )
    top_arms = all_arms_sorted[: args.num_arms]
    print("\nUsing these top arms:")
    for arm in top_arms:
        print(" ", arm)

    # 2) Build subset
    build_subset(
        file_path=args.file_path,
        output_path=args.output_path,
        target_day=args.target_day,
        top_arms=top_arms,
        arm_column=args.arm_column,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
