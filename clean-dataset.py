import pandas as pd

RAW_DATA_PATH = "./car_dataset.csv"
ENCODING = False
NORMALISE = False
INVALID_COLOURS = ["-", "take note", "colorful", "different color"]


cols_to_make_numeric = ["mileage", "num_of_doors", "seating_capacity"]
cols_with_outliers_to_remove = [
    "mileage", 
    "num_of_doors",
    "engine_capacity", 
    "fuel_consumption", 
    "price"
]
cols_to_normalise = [
    "mileage", 
    "engine_capacity", 
    "num_of_doors", 
    "seating_capacity", 
    "fuel_consumption", 
    "price"
]
cols_to_make_lowercase = [
    "origin", 
    "condition", 
    "car_model", 
    "exterior_color", 
    "interior_color", 
    "type_of_engine", 
    "fuel_system", 
    "transmission", 
    "drive_type", 
    "brand", 
    "grade"
]
cols_to_hot_one_encode = [
    "car_model", 
    "exterior_color", 
    "interior_color", 
    "type_of_engine", 
    "fuel_system", 
    "drive_type", 
    "brand", 
    "grade"
]

# Max number of rows / columns displayed in terminal options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv(RAW_DATA_PATH)


#############
# Functions #
#############

# Returns a column with values normalised between 0 and 1.
def normalise_col(col, decimal = 2):
    min_val = col.min()
    max_val = col.max()
    return round(((col - min_val) / (max_val - min_val)), decimal)

# Calculates and returns the lower and upper fence of the given column using IQR.
def calculate_col_iqr(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return lower_fence, upper_fence


##########################
# Section 1 - Formatting #
##########################

# Convert the "mileage", "num_of_doors", and "seating_capacity" columns to integers,
# entries that cannot be converted are changed to NaN.
for col in cols_to_make_numeric:
    df[col] = ["".join(filter(str.isdigit, val)) for val in df[col]]
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Split the "engine" column into "type_of_engine" and "engine_capacity", removing the original column.
df[["type_of_engine", "engine_capacity"]] = df["engine"].str.split("\t",n=1,expand=True)
df = df.drop(columns="engine")

# Convert the "engine_capacity" column to float.
df["engine_capacity"] = df["engine_capacity"].str.lower().str.removesuffix("l").astype(float)

# Convert the "fuel_consumption" column to float, 
# entries that cannot be converted are changed to NaN.
df["fuel_consumption"] = df["fuel_consumption"].str.split("\t").str[0]
df["fuel_consumption"] = df["fuel_consumption"].str.lower().str.removesuffix("l").str.removesuffix(",")
df["fuel_consumption"] = pd.to_numeric(df["fuel_consumption"], errors="coerce")

# Rename the "price (AUD)" column to "price", for consistency. 
df = df.rename(columns={"price (AUD)": "price"})

# Format the "price" column into float, rounded to 2 decimal places.
df["price"] = df["price"].round(2)

# Convert columns to lowercase
for col in cols_to_make_lowercase:
    df[col] = df[col].str.lower()

# Encode columns
if ENCODING:
    encoded_df = pd.get_dummies(df, columns=cols_to_hot_one_encode, dtype=int)
    encoded_df["origin"] = encoded_df["origin"].map({"domestic assembly": 0, "imported": 1})
    encoded_df["condition"] = encoded_df["condition"].map({"used car": 0, "new car": 1})
    encoded_df["transmission"] = encoded_df["transmission"].map({"manual": 0, "automatic": 1})
    encoded_df.to_csv("encoded_car_dataset.csv", index=False)

# Normaise columns
if NORMALISE:
    for col in cols_to_normalise:
        df["normalised_" + col] = normalise_col(df[col], 2)


####################################
# Section 2 - Cleaning & Exploring #
####################################

# Set fuel consumption and engine capacity values for electric vehicles to 0.0.
df["fuel_consumption"] = df["fuel_consumption"].where(df["type_of_engine"] != "electric", 0.0)
df["engine_capacity"] = df["engine_capacity"].where(df["type_of_engine"] != "electric", 0.0)

# Clean the "exterior_color" and "interior_color" columns
df[["exterior_color", "interior_color"]] = df[["exterior_color", "interior_color"]].replace(INVALID_COLOURS, "unknown")
df[["exterior_color", "interior_color"]] = df[["exterior_color", "interior_color"]].replace("gray", "grey")

# Fill empty "fuel_system", "fuel_consumption", "engine_capacity", "year_of_manufacture" columns with "unknown".
df[["fuel_system", "fuel_consumption", "engine_capacity", "year_of_manufacture"]] = df[[
    "fuel_system", "fuel_consumption", "engine_capacity", "year_of_manufacture"]].fillna("unknown")

for col in cols_with_outliers_to_remove:
    lower, upper = calculate_col_iqr(df[col])
    df = df[df[col].between(lower, upper, inclusive="both")]

df.to_csv("cleaned_car_dataset.csv", index=False)