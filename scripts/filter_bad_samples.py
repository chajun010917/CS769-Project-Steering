#!/usr/bin/env python3
"""Remove data points where bad_sample is true from a JSON file."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to output JSON file (defaults to input path with .filtered.json)",
    )
    args = parser.parse_args()

    input_path = args.input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = input_path.with_suffix(".filtered.json")

    # Read the JSON file
    print(f"Reading from: {input_path}")
    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    # Check if it's a list
    if not isinstance(data, list):
        print(f"Error: Expected JSON array, got {type(data).__name__}")
        return

    # Filter out entries where bad_sample is True
    total_count = len(data)
    filtered_data = []
    removed_count = 0

    for entry in data:
        if isinstance(entry, dict):
            bad_sample = entry.get("bad_sample", False)
            if bad_sample is True:
                removed_count += 1
                continue
        filtered_data.append(entry)

    kept_count = len(filtered_data)

    # Write filtered data to output file
    print(f"Writing to: {output_path}")
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(filtered_data, outfile, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete:")
    print(f"  Total samples: {total_count}")
    print(f"  Removed (bad_sample=True): {removed_count}")
    print(f"  Kept: {kept_count}")
    print(f"  Output written to: {output_path}")


if __name__ == "__main__":
    main()

