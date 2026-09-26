
import os
import pandas as pd


INPUT_PATH = "data/processed/integrated_features.csv"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    data["trading_date"] = pd.to_datetime(
        data["trading_date"],
        errors="coerce"
    )

    print("=" * 60)
    print("INTEGRATED FEATURE DATASET AUDIT")
    print("=" * 60)

    # -----------------------------------
    # 1. DATASET OVERVIEW
    # -----------------------------------

    print("\n========== DATASET OVERVIEW ==========")

    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    print("\nColumn names:")
    print(data.columns.tolist())

    # -----------------------------------
    # 2. DATE COVERAGE
    # -----------------------------------

    print("\n========== DATE COVERAGE ==========")

    print("Start date:", data["trading_date"].min())
    print("End date:", data["trading_date"].max())

    print(
        "Unique trading dates:",
        data["trading_date"].nunique()
    )

    print(
        "Unique assets:",
        data["asset_id"].nunique()
    )

    # -----------------------------------
    # 3. MISSING VALUES
    # -----------------------------------

    print("\n========== MISSING VALUES ==========")

    missing_counts = data.isnull().sum()

    missing_counts = missing_counts[
        missing_counts > 0
    ].sort_values(ascending=False)

    if missing_counts.empty:
        print("No missing values found.")

    else:
        print(missing_counts)

    # -----------------------------------
    # 4. MISSING PERCENTAGES
    # -----------------------------------

    print("\n========== MISSING PERCENTAGES ==========")

    missing_percentage = (
        data.isnull().mean() * 100
    ).round(2)

    missing_percentage = missing_percentage[
        missing_percentage > 0
    ].sort_values(ascending=False)

    if missing_percentage.empty:
        print("No missing values found.")

    else:
        print(missing_percentage)

    # -----------------------------------
    # 5. FACTOR COVERAGE
    # -----------------------------------

    print("\n========== FACTOR COVERAGE ==========")

    factor_columns = [
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
        "rf"
    ]

    available_factor_columns = [
        column
        for column in factor_columns
        if column in data.columns
    ]

    if not available_factor_columns:
        print("No factor columns found.")

    else:
        print("\nMissing factor values:")

        for column in available_factor_columns:
            print(
                f"{column}:",
                data[column].isnull().sum()
            )

        complete_factors = data.dropna(
            subset=available_factor_columns
        )

        if complete_factors.empty:
            print(
                "\nNo rows contain complete factor data."
            )

        else:
            print(
                "\nFirst complete factor date:",
                complete_factors["trading_date"].min()
            )

            print(
                "Last complete factor date:",
                complete_factors["trading_date"].max()
            )

            print(
                "Rows with complete factors:",
                len(complete_factors)
            )

            print(
                "Assets with complete factors:",
                complete_factors["asset_id"].nunique()
            )

    # -----------------------------------
    # 6. MACRO COVERAGE
    # -----------------------------------

    print("\n========== MACRO COVERAGE ==========")

    macro_columns = [
        "CPI",
        "FEDERAL_FUNDS_RATE",
        "TREASURY_10Y",
        "UNEMPLOYMENT_RATE",
        "VIX"
    ]

    available_macro_columns = [
        column
        for column in macro_columns
        if column in data.columns
    ]

    if not available_macro_columns:
        print("No macro columns found.")

    else:
        print("\nMissing macro values:")

        for column in available_macro_columns:
            print(
                f"{column}:",
                data[column].isnull().sum()
            )

        print("\nMacro date coverage:")

        for column in available_macro_columns:
            available_macro = data.dropna(
                subset=[column]
            )

            if available_macro.empty:
                print(
                    f"{column}: No available observations"
                )

            else:
                print(
                    f"{column}:",
                    available_macro["trading_date"].min(),
                    "to",
                    available_macro["trading_date"].max()
                )

    # -----------------------------------
    # 7. DUPLICATES
    # -----------------------------------

    print("\n========== DUPLICATES ==========")

    print(
        "Duplicate complete rows:",
        data.duplicated().sum()
    )

    print(
        "Duplicate asset-date rows:",
        data.duplicated(
            subset=["asset_id", "trading_date"]
        ).sum()
    )

    # -----------------------------------
    # 8. FEATURE DATA TYPES
    # -----------------------------------

    print("\n========== DATA TYPES ==========")

    print(data.dtypes)

    # -----------------------------------
    # 9. FINAL SUMMARY
    # -----------------------------------

    print("\n========== AUDIT SUMMARY ==========")

    print("Dataset shape:", data.shape)

    print(
        "Total missing values:",
        data.isnull().sum().sum()
    )

    print(
        "Total duplicate rows:",
        data.duplicated().sum()
    )

    print("\nAudit completed successfully.")


if __name__ == "__main__":
    main()