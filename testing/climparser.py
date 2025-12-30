import pandas as pd

def load_dataset(filename):
    climrr_df = pd.read_csv("FullData.csv")
    return climrr_df

def query_center(df, county, state):
    # Filter rows for Stephens County, OK
    location_df = df[
        (df["NAME"].str.strip().str.lower() == county.strip().lower())
        & (df["State_Abbr"].str.strip().str.lower() == state.strip().lower())
    ].copy()

    # Extract row (R###) and column (C###) numbers from 'Crossmodel'
    location_df["row"] = location_df["Crossmodel"].str.extract(r"R(\d+)").astype(int)
    location_df["col"] = location_df["Crossmodel"].str.extract(r"C(\d+)").astype(int)

    # Compute the mean row and column
    center_row = round(location_df["row"].mean())
    center_col = round(location_df["col"].mean())

    # Construct the center Crossmodel string
    center_crossmodel = f"R{center_row:03d}C{center_col:03d}"
    center_row_df = df[df["Crossmodel"] == center_crossmodel]

    return center_row_df.squeeze()


def query_mean(df, county, state):
    # Filter rows by county and state abbreviation (case insensitive)
    location_df = df[
        (df["NAME"].str.strip().str.lower() == county.strip().lower())
        & (df["State_Abbr"].str.strip().str.lower() == state.strip().lower())
    ].copy()

    if location_df.empty:
        print(f"No data found for {county}, {state}")
        return None

    # Compute the mean for all numeric columns
    mean_values = location_df.mean(numeric_only=True)

    # Optional: attach location metadata
    mean_values["county"] = county.title()
    mean_values["state"] = state.upper()
    mean_values["num_cells"] = len(location_df)

    return mean_values
