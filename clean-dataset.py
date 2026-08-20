import pandas as pd

RAW_DATA_PATH = "./car_dataset.csv"
ENCODING = True
NORMALISE = True
INVALID_COLOURS = ["-", "take note", "colorful", "different color"]


cols_to_make_numeric = ["mileage", "num_of_doors", "seating_capacity"]
cols_to_check_for_outliers = [
    "mileage", 
    "num_of_doors",
    "engine_capacity", 
    "fuel_consumption", 
    "price (AUD)"
]
cols_to_normalise = [
    "mileage", 
    "engine_capacity", 
    "num_of_doors", 
    "seating_capacity", 
    "fuel_consumption", 
    "price (AUD)"
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
cols_to_calculate_statistics = [
    "price (AUD)", 
    "mileage", 
    "year_of_manufacture"
]

# Max number of rows / columns displayed in terminal options
pd.set_option('display.max_rows', 50)
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

def filter_outliers_in_col(col, lower, upper):
    return((col >= upper) | (col <= lower))


    


##########################
# Section 1 - Formatting #
##########################

# Convert the "mileage", "num_of_doors", and "seating_capacity" columns to integers,
# entries that cannot be converted are changed to NaN.
for col in cols_to_make_numeric:
    df[col] = ["".join(filter(str.isdigit, val)) for val in df[col]]
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert the "fuel_consumption" column to float, 
# entries that cannot be converted are changed to NaN.
df["fuel_consumption"] = df["fuel_consumption"].str.split("\t").str[0]
df["fuel_consumption"] = df["fuel_consumption"].str.lower().str.removesuffix("l").str.removesuffix(",")
df["fuel_consumption"] = pd.to_numeric(df["fuel_consumption"], errors="coerce")

# Format the "price (AUD)" column into float, rounded to 2 decimal places.
df["price (AUD)"] = df["price (AUD)"].astype(float).round(2)

# Split the "engine" column into "type_of_engine" and "engine_capacity", removing the original column.
df[["type_of_engine", "engine_capacity"]] = df["engine"].str.split("\t",n=1,expand=True)
df = df.drop(columns="engine")

# Convert the "engine_capacity" column to float.
df["engine_capacity"] = df["engine_capacity"].str.lower().str.removesuffix("l").astype(float)

# Encode columns
# Encoding is currently performed on a separate data frame due to 
# the task 1d requiring the original columns to remain as strings
# Waiting for clarification from course convenor before making any changes 
if ENCODING:
    encoded_df = pd.get_dummies(df, columns=cols_to_hot_one_encode, dtype=int)
    encoded_df["origin"] = encoded_df["origin"].map({"Domestic assembly": 0, "Imported": 1})
    encoded_df["condition"] = encoded_df["condition"].map({"Used car": 0, "New car": 1})
    encoded_df["transmission"] = encoded_df["transmission"].map({"Manual": 0, "Automatic": 1})
    encoded_df.to_csv("encoded_car_dataset.csv", index=False)

# Convert columns to lowercase
for col in cols_to_make_lowercase:
    df[col] = df[col].str.lower()

# Normaise columns
if NORMALISE:
    for col in cols_to_normalise:
        df["normalised_" + col] = normalise_col(df[col])


####################################
# Section 2 - Cleaning & Exploring #
####################################

# Calculate mean, median, mode, standard deviation, and range of "price (AUD)", "mileage", and "year_of_manufacture"
statistics = {}

for col in cols_to_calculate_statistics:
    statistics[col] = {
        "mean" : round(df[col].mean(), 2),
        "median" : round(df[col].median(), 2),
        "mode" : round(df[col].mode()[0], 2),
        "standard_deviation" : round(df[col].std(), 2),
        "range" : round((df[col].max() - df[col].min()), 2)
    }



for col in cols_to_check_for_outliers:
    lower, upper = calculate_col_iqr(df[col])
    outlier_mask = filter_outliers_in_col(df[col], lower, upper)
    outliers = df.loc[outlier_mask, col]

    print(
        f"\nColumn: {col}"
        f"\nLower fence: {lower}"
        f"\nUpper fence: {upper}"
        f"\nNumber of outliers: {len(outliers)}"
    )
    unique_outliers = outliers.unique()
    print(
        f"\nNumber of unique outliers: {len(unique_outliers)}"
        f"\nUnique outliers: {pd.Series(unique_outliers).sort_values(ascending=True).to_numpy()}"
    )


# mileage - All outliers are above the upper fence, though the upper fence is a reasonable value for mileage on a vehicle.
# Though unlikely, it is possible for a vehicle to have a mileage of 1,000,000km. Due to the statistic improbability of 
# this many vehicles having over 1,000,000km mileage, all values over 1,000,000 have been removed.




# num_of_doors - The lower fence is 2.5, but it is possible for a vehicle to have 1 door e.g., a bus. It is also possible, though unlikely,
# for a vehicle to have 7 doors. However, 42 and above is likely an impossible value to achieve.
# Therefore values between 1 and 7 were kept, due to being possible values, even if unlikely.
# Values below 1 and above 7 have been removed due to their likely impossible nature. 

# engine_capacity - The engine capacity value of electric vehicles have been set to 0, as they do not have an engine capaicty.
# The lower fence for engine capacity is 0.1. However, an engine capacity of 0.1 or below is considered unlikely for a non-electric vehicle.
# Therefore, any capacity 0.1 or lower has been removed.
# Manual inspection has revealed that one non-electric vehicle is listed as having an engine capacity of 0.2L.
# This is much lower than the next closest non-electric vehicles engine capacity of 0.8L. As a result, the anomolous entry of 0.2 has also been removed.
# The upper fence is 3.75L, however, many trucks have an engine capacity of 13 litres. 
# As the highest valued outlier is 12.7 litres, values above the upper fence can remain.

df["engine_capacity"] = df["engine_capacity"].where(df["type_of_engine"] != "electric", 0.0)


# fuel_consumption - some outliers above the upper fence are reasonable, while others verge on unlikely or impossible. 
# The only value below the lower fence that could be considered unrealistic is 0, which is entirely achieveable in electric vehicles.
# Therefore, values below the lower fence will remain, values above 20 will be removed.

df["fuel_consumption"] = df["fuel_consumption"].where(df["type_of_engine"] != "electric", 0.0)


# price (AUD) - The IQR upper fence is $76,518.37, but this is a reasonable price for many vehices.
# Manual inspection of outliers with values above the upper fence + standard deviation revealed that most vehicles
# within that price range are luxury vehicles and it is therefore reasonable to conclude that the listed prices are accurate, 
# as a result, no outliers will be removed. However, manual inspection also reveals that many values are near-zero, which is far above the lower fence,
# but still an unlikely price for a vehicle. The lowest price that is not near-zero is $728.75, which is a valid price for a vehice.
# The difference between 728.75 and the near-zero values is a clear separation indicative of anomalous data, therefore all values below 700 will be removed.




# Set fuel consumption and engine capacity values for electric vehicles to 0.0.



# Fill empty "fuel_system" entries with "unknown".
#df[["fuel_system"]] = df[["fuel_system"]].fillna("unknown")

# Clean the "exterior_color" and "interior_color" columns
#df[["exterior_color", "interior_color"]] = df[["exterior_color", "interior_color"]].replace(INVALID_COLOURS, "unknown")
#df[["exterior_color", "interior_color"]] = df[["exterior_color", "interior_color"]].replace("gray", "grey")



#df.to_csv("cleaned_car_dataset.csv", index=False)