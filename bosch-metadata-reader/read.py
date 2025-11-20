#!/usr/bin/env python3

INPUT_FILE = "combined_vehicle_stats_expanded.csv"
OUTPUT_FILE = "output.csv"
LIMIT = 10000   # number of lines to keep (including header)

def main():
    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            outfile.write(line)
            count += 1

            if count >= LIMIT + 1:   # +1 because header counts as the first line
                break

    print(f"✅ Created '{OUTPUT_FILE}' with first {LIMIT} lines from '{INPUT_FILE}'.")

if __name__ == "__main__":
    main()
