import pandas as pd
import re

def attach(df: pd.DataFrame, xl: pd.ExcelFile) -> pd.DataFrame:
    # Parse first sheet
    first_sheet = xl.sheet_names[0]
    ref_df = xl.parse(first_sheet)

    # Find datetime column (case-insensitive)
    datetime_col = None
    for col in ref_df.columns:
        if str(col).lower() in ["datetime", "dateTime".lower(), "time", "date"]:
            datetime_col = col
            break

    if datetime_col is None:
        raise ValueError(f"No datetime-like column found in {first_sheet}")

    # Standardize datetime format
    ref_df = ref_df.rename(columns={datetime_col: "Datetime"})
    ref_df["Datetime"] = pd.to_datetime(ref_df["Datetime"], errors="coerce")

    # Keep only Datetime + close
    if "close" not in ref_df.columns:
        raise ValueError(f"'close' column not found in {first_sheet}")
    ref_df = ref_df[["Datetime", "close"]].dropna()

    # Also standardize in main df
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # Merge on Datetime (inner join keeps only matches)
    merged = pd.merge(df, ref_df, on="Datetime", how="left")

    return merged


def convert(file_path: str) -> pd.DataFrame:
    """Convert Excel sheets into a stacked DataFrame of datetime lookups."""
    xl = pd.ExcelFile(file_path)
    
    # First sheet (reference)
    first_sheet = xl.sheet_names[1]
    df_first = xl.parse(first_sheet)
    
    if "dateTime" not in df_first.columns:
        raise ValueError(f"'dateTime' column not found in {first_sheet}")
    
    datetime_values = df_first["dateTime"].dropna().unique()
    
    results_per_datetime = []
    
    for dt in datetime_values:
        matches = []
        
        for sheet in xl.sheet_names[1:]:
            df = xl.parse(sheet)
            
            if "dateTime" not in df.columns:
                continue
            
            matched = df[df["dateTime"] == dt].copy()
            
            if not matched.empty:
                cols_to_keep = [c for c in ["close_ce", "close_pe"] if c in matched.columns]
                subset = matched[cols_to_keep].copy()
                
                subset.insert(0, "Datetime", dt)
                strike_val = re.search(r"(\d+)$", sheet).group(1)
                subset.insert(1, "strike", float(strike_val))
                
                matches.append(subset)
        
        if matches:
            combined = pd.concat(matches, ignore_index=True)
            results_per_datetime.append(combined)
    
    if results_per_datetime:
        df = pd.concat(results_per_datetime, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["Datetime", "strike", "close_ce", "close_pe"])

    final_df = attach(df=df, xl=xl)
    final_df = final_df.rename(columns={
        "close_ce": "call",
        "close_pe": "put",
        "close":"underlying"
    })

    # Reorder
    order = ['Datetime', 'underlying', 'strike', 'call', 'put']
    final_df = final_df[[c for c in order if c in final_df.columns]]
    return final_df


def save_output(df: pd.DataFrame, save_path: str):
    """Save DataFrame to Excel or CSV depending on file extension."""
    if save_path.endswith(".xlsx"):
        df.to_excel(save_path, index=False)
        print("Saved")
    elif save_path.endswith(".csv"):
        df.to_csv(save_path, index=False)
        print("Saved")
    else:
        raise ValueError("Save path must end with .xlsx or .csv")

def split(df: pd.DataFrame):
    # Ensure sorted by datetime
    df = df.sort_values("Datetime")
    
    # Group into a list of DataFrames
    grouped_list = [group for _, group in df.groupby("Datetime")]
    
    return grouped_list