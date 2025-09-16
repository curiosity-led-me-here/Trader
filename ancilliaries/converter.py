import pandas as pd

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
                subset.insert(1, "SourceSheet", sheet)
                
                matches.append(subset)
        
        if matches:
            combined = pd.concat(matches, ignore_index=True)
            results_per_datetime.append(combined)
    
    if results_per_datetime:
        return pd.concat(results_per_datetime, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Datetime", "SourceSheet", "close_ce", "close_pe"])


def save_output(df: pd.DataFrame, save_path: str):
    """Save DataFrame to Excel or CSV depending on file extension."""
    if save_path.endswith(".xlsx"):
        df.to_excel(save_path, index=False)
    elif save_path.endswith(".csv"):
        df.to_csv(save_path, index=False)
    else:
        raise ValueError("Save path must end with .xlsx or .csv")

def split(df: pd.DataFrame):
    # Ensure sorted by datetime
    df = df.sort_values("Datetime")
    
    # Group into a list of DataFrames
    grouped_list = [group for _, group in df.groupby("Datetime")]
    
    return grouped_list

path = r'/Users/ashu/Documents/Trader/data/NIFTY 51/2024-11-28_16-38-19.xlsx'

data = convert(path)
print(data)