####################################################################################################
# Some tasks in this assignment have been reorganised in order to support                          #
# a more efficient and logical EDA pipeline. Below is a list of changes made to a tasks order.     #
#                                                                                                  #
# - Task 1c has been moved to be completed after 2b. This has been done to ensure that all         #
# encoded columns have the same capitalisation, instead of having multiple encoded columns         #
# that mean the same thing with different capitalisation.                                          #
#                                                                                                  #
# - Task 1e has been moved to be completed after 2b. This has been done to ensure that all         #
# normalisation is completed after cleaning of outliers and missing values, ensuring the           #
# normalised values will accurately reflect the data, rather than being compressed.                #
#                                                                                                  #
# - Task 2a is initially completed before 2b to assist in exploring the uncleaned data and aid     #
# in discovering any anomalies that may be present. Task 2a is completed again after completion    #
# of 2b, as to demonstrate how unclean data can impact insights gained from the EDA process.       #
#                                                                                                  #
# Additionally, changes to a tasks order will be identified in a comment before the code block     #
# pertaining to the task in question.                                                              #
####################################################################################################

import pandas as pd

RAW_DATA_PATH = "./car_dataset.csv"
ENCODING = False
NORMALISE = False

df = pd.read_csv(RAW_DATA_PATH)

# Rename the "price (AUD)" column to "price_aud", for consistency.
df = df.rename(columns = {"price (AUD)" : "price_aud"})

INVALID_COLOURS = [
    "-", 
    "take note", 
    "colorful", 
    "different color"
]

COLS_TO_MAKE_NUMERIC = [
    "mileage", 
    "num_of_doors", 
    "seating_capacity"
]

COLS_TO_CHECK_FOR_OUTLIERS = [
    "mileage", 
    "num_of_doors",
    "engine_capacity", 
    "fuel_consumption", 
    "price_aud",
    "seating_capacity"
]

COLS_TO_NORMALISE = [
    "mileage", 
    "engine_capacity", 
    "num_of_doors", 
    "seating_capacity", 
    "fuel_consumption", 
    "price_aud"
]

COLS_TO_MAKE_LOWERCASE = [
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

COLS_TO_HOT_ONE_ENCODE = [
    "car_model", 
    "exterior_color", 
    "interior_color", 
    "type_of_engine", 
    "fuel_system", 
    "drive_type", 
    "brand", 
    "grade"
]

COLS_TO_CALCULATE_STATISTICS = [
    "price_aud", 
    "mileage", 
    "year_of_manufacture"
]


#############
# Functions #
#############

# Returns a column with values normalised between 0 and 1.
def normalise_col(column : pd.Series, precision = 2):
    min_val = column.min()
    max_val = column.max()
    return round(((column - min_val) / (max_val - min_val)), precision)

# Calculates and returns the lower and upper fence of the given column using IQR.
def calculate_col_iqr(column : pd.Series):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)
    return lower_fence, upper_fence

# Returns a series mask for values outside the IQR in a series.
def get_outlier_mask(column : pd.Series, lower, upper):
    return((column > upper) | (column < lower))

# Prints the name, lower and upper IQR fences, number of outliers, number of unique outliers,
# and an numpy array of unique outliers for a specific column, column_name.
def display_col_outlier_results(column_name : str, outliers_results : dict):
    print(
        f"\nColumn: {column_name}"
        f"\nLower fence: {outliers_results[column_name]["lower_fence"]}"
        f"\nUpper fence: {outliers_results[column_name]["upper_fence"]}"
        f"\nNumber of outliers: {len(outliers_results[column_name]["outliers"])}"
    )
    unique_outliers = outliers_results[column_name]["outliers"].unique()
    print(
        f"\nNumber of unique outliers: {len(unique_outliers)}"
        f"\nUnique outliers: {pd.Series(unique_outliers).sort_values(ascending=True).to_numpy()}"
    )

###############################
# Section 1 - Data Formatting #
###############################
# Task 1A #
###########

# Convert the "mileage", "num_of_doors", and "seating_capacity" columns to integers,
# entries that cannot be converted are changed to NaN.
for column_name in COLS_TO_MAKE_NUMERIC:
    df[column_name] = ["".join(filter(str.isdigit, val)) for val in df[column_name]]
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

# Convert the "fuel_consumption" column to float, 
# entries that cannot be converted are changed to NaN.
df["fuel_consumption"] = df["fuel_consumption"].str.split("\t").str[0]
df["fuel_consumption"] = df["fuel_consumption"].str.lower().str.removesuffix("l").str.removesuffix(",")
df["fuel_consumption"] = pd.to_numeric(df["fuel_consumption"], errors="coerce")

# Format the "price_aud" column into float, rounded to 2 decimal places.
df["price_aud"] = df["price_aud"].astype(float).round(2)


###########
# Task 1B #
###########

# Split the "engine" column into "type_of_engine" and "engine_capacity", removing the original column.
df[["type_of_engine", "engine_capacity"]] = df["engine"].str.split("\t",n=1,expand=True)
df = df.drop(columns="engine")

# Convert the "engine_capacity" column to float.
df["engine_capacity"] = df["engine_capacity"].str.lower().str.removesuffix("l").astype(float)


#################################################
# Task 1C - Moved to be completed after Task 2B #
#################################################

###########
# Task 1D #
###########

# Convert columns to lowercase
for column_name in COLS_TO_MAKE_LOWERCASE:
    df[column_name] = df[column_name].str.lower()


#################################################
# Task 1E - Moved to be completed after Task 2B #
#################################################


#############################################
# Section 2 - Data Exploration and Cleaning #
#############################################
# Task 2A #
###########

# Calculate mean, median, mode, standard deviation, and range of "price_aud", "mileage", and "year_of_manufacture"
statistics = {}
for column_name in COLS_TO_CALCULATE_STATISTICS:
    statistics[column_name] = {
        "mean" : round(df[column_name].mean(), 2),
        "median" : round(df[column_name].median(), 2),
        "mode" : round(df[column_name].mode()[0], 2),
        "standard_deviation" : round(df[column_name].std(), 2),
        "range" : round((df[column_name].max() - df[column_name].min()), 2)
    }


#############
# Task 2B i #
#############

# Detect outliers using the IQR method.
outlier_results = {}
for column_name in COLS_TO_CHECK_FOR_OUTLIERS:
    lower, upper = calculate_col_iqr(df[column_name])
    outlier_mask = get_outlier_mask(df[column_name], lower, upper)
    outlier_results[column_name] = {
        "lower_fence" : lower, 
        "upper_fence" : upper,
        "outliers" : df.loc[outlier_mask, column_name]
    }
    

########################################
# Justifications for outlier treatment #
########################################

display_col_outlier_results("mileage", outlier_results)
df = df[(df["mileage"] <= 1000000)]

# mileage - The upper IQR fence is 150,000Km. While all outliers are above the upper IQR fence,
# it is considered reasonable value for mileage on a vehicle. Some entries contain outlier 
# values exceeding 1,000,000Km, which sounds erroneous. However, it is possible for 
# a vehicle to have a mileage of 1,000,000Km. Though, it is statistically unlikely that
# this dataset would contain this many vehicles that have over 1,000,000Km mileage.
# Therefore, all values over 1,000,000 are considered erroneous and have been removed.


display_col_outlier_results("num_of_doors", outlier_results)
df = df[df["num_of_doors"].between(1, 7, inclusive = "both")]

# num_of_doors - The lower IQR fence is 2.5, but it is possible for a vehicle to 
# have 1 door e.g., a bus. It is also possible, though uncommon, for a vehicle 
# to have 7 doors. However, 42+ doors is clearly erroneous. Therefore values 
# between 1 and 7 were kept, due to being possible values, even if unlikely. 
# Values below 1 and above 7 are considered erroneous and have been removed.


display_col_outlier_results("engine_capacity", outlier_results)
df = df[~((df["engine_capacity"] == 0.2) & (df["type_of_engine"] == "petrol"))]

# engine_capacity - The lower IQR fence for engine capacity is 0.15L. 
# Many electric vehicle entries have an engine capacity of 0.1L, however,
# electric vehicles do not have an internal-combustion engine and therefore
# do not have an engine capacity. These values associated with electric vehicles 
# have been retained for now, and will instead be transformed in Task 2B ii.
# Manual inspection revealed that one non-electric vehicle is listed as 
# having an engine capacity of 0.2L. This is substantially lower than the other
# non-electric vehicles represented in the dataset. Therefore, the value is 
# considered erroneous, and has been removed.
# The upper IQR fence is 3.75L. However, engine capacities above this value are 
# possible for large vehicles, such as trucks. Manual inspection has revealed that
# outliers above the upper IQR fence are associated with vehicles that could plausibly
# have a large engine capacity, including the highest valued entry of 12.7 litres.
# Therefore, values above the upper IQR fence have been retained.

display_col_outlier_results("fuel_consumption", outlier_results)
df = df[(df["fuel_consumption"] <= 20.0) | (df["fuel_consumption"].isna())]


# fuel_consumption - Some outliers above the upper IQR fence are reasonable, while others verge on unlikely or impossible. 
# The only value below the lower fence that could be considered unrealistic is 0, which is entirely achieveable in electric vehicles.
# Therefore, values below the lower fence will remain, values above 20 will be removed.

display_col_outlier_results("price_aud", outlier_results)
df = df[df["price_aud"] >= 700]


# price_aud - The IQR upper fence is $76,518.37, but this is a reasonable price for many vehices.
# Manual inspection of outliers with values above the upper fence + standard deviation revealed that most vehicles
# within that price range are luxury vehicles and it is therefore reasonable to conclude that the listed prices are accurate, 
# as a result, no outliers will be removed. However, manual inspection also reveals that many values are near-zero, which is far above the lower fence,
# but still an unlikely price for a vehicle. The lowest price that is not near-zero is $728.75, which is a valid price for a vehice.
# The difference between 728.75 and the near-zero values is a clear separation indicative of anomalous data, therefore all values below 700 will be removed.

##############
# Task 2B ii #
##############################################################
# Justifications for missing and/or invalid values treatment #
##############################################################
print(df.isna().sum())
print("Missing fuel_system entries:", df["fuel_system"].isna().sum())
# Fill empty "fuel_system" entries with "unknown".
df[["fuel_system"]] = df[["fuel_system"]].fillna("unknown")
print("Missing fuel_system entries:", df["fuel_system"].isna().sum())

# fuel_system - Many of the fuel system values are missing. 
# When purchasing a vehice, knowing the fuel system may be important.
# It would be very bad if the fuel system advertised is different to 
# the one actually installed in the vehicle. As there is not enough
# information to accurately impute a value with little-to-no doubt,
# unknown values have been replaced with "unknown"

print("Missing engine_capacity entries:", df["engine_capacity"].isna().sum())
# Set fuel consumption and engine capacity values for electric vehicles to 0.0.

#df["engine_capacity"] = df["engine_capacity"].where(df["type_of_engine"] != "electric", 0.0)

# engine_capacity - Many of the engine capacity values are above 0
# in entries where the type of engine is electric. Engine capacity represents 
# the displacement of an internal combustion engine, measured in litres. 
# Non-hybrid electric vehicles do not have an internal combustion engine and therefore 
# have no engine displacement. As such, the engine capacity value of electric vehicles has been set to 0.

print("Missing fuel_consumption entries:", df["fuel_consumption"].isna().sum())

print("Number of entries where the fuel_consumption of an electric vehicle is not 0.0:",
    ((df["type_of_engine"] == "electric") & ((df["fuel_consumption"] > 0.0) | df["fuel_consumption"].isna())).sum())

df["fuel_consumption"] = df["fuel_consumption"].where(df["type_of_engine"] != "electric", 0.0)

print("Number of entries where the fuel_consumption of an electric vehicle is not 0.0:",
    ((df["type_of_engine"] == "electric") & ((df["fuel_consumption"] > 0.0) | df["fuel_consumption"].isna())).sum())

# fuel_consumption - Similarly to engine capacity, many of the values in 
# fuel consumption are above 0 in entries where the type of engine is electric.
# As previously discussed, electric engines do not consume liquid fuel and therefore do not
# have a fuel consumption rate. As such, the fuel consumption value of electric vehicles
# has been set to 0.

print("Missing year_of_manufacture entries:", df["year_of_manufacture"].isna().sum())

# year_of_manufacture - Manual inspection of the 32 missing values in year of manufacture
# revealed that each of these entries has a note in the car_name column, indicating that 
# the vehicle was manufactured "Before 1990". 

print("Missing exterior_color entries:", df["exterior_color"].isna().sum())
print("Missing interior_color entries:", df["interior_color"].isna().sum())
# Clean the "exterior_color" and "interior_color" columns
df[["exterior_color", "interior_color"]] = df[["exterior_color", "interior_color"]].replace(INVALID_COLOURS, "unknown")
df[["exterior_color", "interior_color"]] = df[["exterior_color", "interior_color"]].replace("gray", "grey")

# exterior_color and interior_color - These columns have both been given the same treatment,
# as they are closely related and contain many overlapping values. Many values in these columns
# are not considered to be colours. Towards the beginning of this file, there is a constant list named
# "INVALID_COLOURS", containing colours that are considered to be invalid. As these colours are not
# descriptive enough to convey useful information to someone reading the data, they have been replacedwith "unknown". 
# Additionally, the American spelling of "gray" has been replaced with the English spelling, "grey".

print(df.isna().sum())
###############################################
# Task 2A - Intentionally duplicated after 2B #
###############################################

# Calculate mean, median, mode, standard deviation, and range of "price_aud", "mileage", and "year_of_manufacture"
statistics = {}
for column_name in COLS_TO_CALCULATE_STATISTICS:
    statistics[column_name] = {
        "mean" : round(df[column_name].mean(), 2),
        "median" : round(df[column_name].median(), 2),
        "mode" : round(df[column_name].mode()[0], 2),
        "standard_deviation" : round(df[column_name].std(), 2),
        "range" : round((df[column_name].max() - df[column_name].min()), 2)
    }

###########
# Task 1C #
###########

# Encode columns
if ENCODING:
    for column_name in COLS_TO_HOT_ONE_ENCODE:
        hot_one_encoded_cols = pd.get_dummies(df[column_name], prefix = column_name, dtype = int)
        df = pd.concat([df, hot_one_encoded_cols], axis = 1)
    df["label_encoded_origin"] = df["origin"].map({"domestic assembly": 0, "imported": 1})
    df["label_encoded_condition"] = df["condition"].map({"used car": 0, "new car": 1})
    df["label_encoded_transmission"] = df["transmission"].map({"manual": 0, "automatic": 1})


###########
# Task 1E #
###########

# Normaise columns, creating a new column for each.
if NORMALISE:
    for column_name in COLS_TO_NORMALISE:
        df["normalised_" + column_name] = normalise_col(df[column_name])


############################
# Section 3 - Data Storage #
############################

# Save processed data to CSV.
df.to_csv("processed_car_detail_en.csv", index=False)