import pandas as pd

def main():
    print("📥 Loading CSV...")
    df = pd.read_csv("combined_vehicle_stats_expanded.csv")

    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print("🔎 Counting timestamps per object_id...")
    counts = df.groupby("object_id")["timestamp"].nunique()

    # objects with more than 1 timestamp (real tracking)
    multi = counts[counts > 1]

    print("\n===========================================")
    print(" OBJECTS WITH MORE THAN ONE TIMESTAMP")
    print("===========================================\n")
    if len(multi) == 0:
        print("❌ No objects appear in more than one timestamp.")
    else:
        print(multi)

    print("\n===========================================")
    print(" SUMMARY")
    print("===========================================\n")
    print(f"Total unique object_ids: {counts.size}")
    print(f"Objects with >1 timestamp: {len(multi)}")
    print(f"Percentage tracked: {(len(multi)/counts.size)*100:.2f}%")

    print("\nDone.")

if __name__ == "__main__":
    main()
