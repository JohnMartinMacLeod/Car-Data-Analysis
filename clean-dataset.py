import pandas as pd

RAW_DATA_PATH = "./car_dataset.csv"

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


df = pd.read_csv(RAW_DATA_PATH)

############################################
##### VALIDATE DATA BEFORE CONVERTING ######
############################################



# Section 1 #
#######################################################################################
print(df)

# Converts a column to an int without any non-numerical characters.
def make_column_numeric(df, col):
    df[col] = ["".join(filter(str.isdigit, val)) for val in df[col]]
    df[col] = df[col].astype(int)

# Returns a column with values normalised between 0 and 1.
def normalise_col(col, decimal = 2):
    min_val = col.min()
    max_val = col.max()
    return round(((col - min_val) / (max_val - min_val)), decimal)

for col in cols_to_make_numeric:
    make_column_numeric(df, col)



# Remove anomalous data entries from "fuel_consumption".
df = df[df["fuel_consumption"].str[0].str.isdigit()]


# Remove non-numeric characters from fuel_consumption and convert from string to integer.
df["fuel_consumption"] = df["fuel_consumption"].str.replace("100Km", "")
df["fuel_consumption"] = ["".join(char for char in val if char.isdigit() or char == ".") for val in df["fuel_consumption"]]
df["fuel_consumption"] = df["fuel_consumption"].astype(float)

# Rename "price (AUD)" to "price" for consistency. 
df = df.rename(columns={"price (AUD)": "price"})

# Split "engine" into "type_of_engine" and "engine_capacity"
df[["type_of_engine", "engine_capacity"]] = df["engine"].str.split(n=1,expand=True)
df = df.drop(columns="engine")

# Delete entries without an engine capacity and clean remaining entries.
df.dropna(subset="engine_capacity", inplace=True)
df["engine_capacity"] = ["".join(char for char in val if char.isdigit() or char == ".") for val in df["engine_capacity"]]
df["engine_capacity"] = df["engine_capacity"].astype(float)



encoded_df = pd.get_dummies(df, columns=cols_to_hot_one_encode, dtype=int)
encoded_df["origin"] = encoded_df["origin"].map({"Domestic assembly": 0, "Imported": 1})
encoded_df["condition"] = encoded_df["condition"].map({"Used car": 0, "New car": 1})
encoded_df["transmission"] = encoded_df["transmission"].map({"Manual": 0, "Automatic": 1})


for col in cols_to_make_lowercase:
    df[col] = df[col].str.lower()

def calculate_col_iqr(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return lower_fence, upper_fence

for col in cols_with_outliers_to_remove:
    lower, upper = calculate_col_iqr(df[col])
    df = df[df[col].between(lower, upper, inclusive="both")]

for col in cols_to_normalise:
    df["normalised_" + col] = normalise_col(df[col], 2)

dfCopy = df.copy()
df.to_csv("cleaned_car_dataset.csv", index=False)

# Section 2 #
#######################################################################################

df = pd.read_csv(RAW_DATA_PATH)

# Rename "price (AUD)" to "price" for consistency. 
df = df.rename(columns={"price (AUD)": "price"})

for col in cols_to_make_numeric:
    make_column_numeric(df, col)

for col in cols_with_outliers_to_remove:
    lower, upper = calculate_col_iqr(df[col])
    df = df[df[col].between(lower, upper)]

def calculate_col_iqr(col):
    min_val = col.min()
    median_val = col.median()
    max_val = col.max()
    q1 = col[col.between(min_val, median_val)].median()
    q3 = col[col.between(median_val, max_val)].median()
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return lower_fence, upper_fence


df.to_csv("cleaned_car_dataset.csv", index=False)