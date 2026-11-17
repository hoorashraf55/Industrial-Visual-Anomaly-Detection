
from src.reporting.data_loader import load_all_results

from src.reporting.table_generator import generate_tables

from src.reporting.chart_generator import generate_charts

from src.reporting.error_analysis import (
    generate_error_analysis_examples
)


def main():

    print("=" * 55)
    print("AI403 Industrial Anomaly Detection Report Generator")
    print("=" * 55)

    # Step 1: Load experiment results

    print("\n[1/4] Loading experiment results...")

    df = load_all_results()

    if df.empty:
        print("No experiment results found.")
        return

    print(f"Loaded {len(df)} experiment result files.")

    # Step 2: Generate report tables

    print("\n[2/4] Generating report tables...")

    all_15_s42 = generate_tables(df)

    # Step 3: Generate charts

    print("\n[3/4] Generating charts...")

    generate_charts(all_15_s42)

    # Step 4: Generate error analysis

    print("\n[4/4] Generating error analysis examples...")

    generate_error_analysis_examples()

    print("\n" + "=" * 55)
    print("Report generation process completed.")
    print("=" * 55)


if __name__ == "__main__":
    main()


