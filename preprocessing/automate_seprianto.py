import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# ================================================================================================
"""START OF DATA LOADING FUNCTION"""
# ================================================================================================
def load_data(file_path='../data/data_student.csv'):
    """Function to load data from a CSV file"""
    
    try:
        data_student = pd.read_csv(file_path, sep=";")
        
        if 'Student_ID' not in data_student.columns:
            data_student.insert(0, 'Student_ID', range(1000001, 1000001 + len(data_student)))
            
            print(data_student.head(5))
            
            return data_student
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
# ================================================================================================
"""END OF DATA LOADING FUNCTION"""
# ================================================================================================


# ================================================================================================
"""START OF PREPROCESSING DATA"""
# ================================================================================================

# Split columns into numerical and categorical features
def get_categorical_features(data):
    """Function to split columns into categorical features"""
    
    try:
        categorical_features = [
                'Marital_status',
                'Application_mode',
                'Course',
                'Daytime_evening_attendance',
                'Previous_qualification',
                'Nacionality',
                'Mothers_qualification',
                'Fathers_qualification',
                'Mothers_occupation',
                'Fathers_occupation',
                'Gender',
                'Age_at_enrollment',
                'Status'
        ]
        return [
            col 
            for col in categorical_features
            if col in data.columns and col != 'Status'
        ]
    
    except Exception as e:
        print(f"An error occurred while getting categorical features: {e}")
        return []


def get_numerical_features(data):
    """Function to split columns into numerical features"""
    try:
        exclusions = ["Student_ID", "Status"]
        
        categorical_features = get_categorical_features(data)
        
        return [
            col for col in data.columns 
            if col not in categorical_features and col not in exclusions
        ]
        
    except Exception as e:
        print(f"An error occurred while getting numerical features: {e}")
        return []


def clean_outlier(data, numerical_features):
    """Function to handle outliers in numerical features using clipping method"""
    try:
        print("\n[1] OUTLIER HANDLING")
        print("-" * 100)

        # Calculate Q1, Q3, and IQR
        Q1 = data[numerical_features].quantile(0.25)
        Q3 = data[numerical_features].quantile(0.75)
        IQR = Q3 - Q1

        # Determine lower and upper bound
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR    

        # Handle outliers using clipping method
        data[numerical_features] = data[numerical_features].clip(lower=lower_bound, upper=upper_bound, axis=1)
        print("Outliers have been handled using clipping method.")

        return data
    
    except Exception as e:
        print(f"An error occurred while handling outliers: {e}")


def standardize_numerical_features(data, numerical_features):
    """Function to standardize numerical features using StandardScaler"""
    try:
        print("\n[2] STANDARDIZING NUMERICAL FEATURES")
        print("-" * 100)
        
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        print("Numerical features have been standardized.")
        
        return data
        
    except Exception as e:
        print(f"An error occurred while standardizing numerical features: {e}")


def encode_status(data):
    """Function to encode the target variable 'Status' using LabelEncoder"""
    try:
        print("\n[3] ENCODING TARGET VARIABLE 'STATUS'")
        print("-" * 100)

        if 'Status' in data.columns:
            le = LabelEncoder()
            data['Status'] = le.fit_transform(data['Status'])
            print("Target variable 'Status' has been encoded.")
        else:
            print("Column 'Status' not found in the dataset.")
        
        return data
    
    except Exception as e:
        print(f"An error occurred while encoding 'Status': {e}")


def binning_categorical_features(data, categorical_features):
    """Function to bin categorical features into categorical""" 
    try:
        print("\n[4] BINNING CATEGORICAL FEATURES")
        print("-" * 100)

        binning_config = {
            "Application_mode": {
                "bins": [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57, 60],
                "labels": [
                    "1st phase - general contingent", "Ordinance No. 612/93", 
                    "1st phase - special contingent (Azores Island)", "Holders of other higher courses", 
                    "Ordinance No. 854-B/99", "International student (bachelor)", 
                    "1st phase - special contingent (Madeira Island)", "2nd phase - general contingent", 
                    "3rd phase - general contingent", "Ordinance No. 533-A/99, item b2) (Different Plan)", 
                    "Ordinance No. 533-A/99, item b3 (Other Institution)", "Over 23 years old", 
                    "Transfer", "Change of course", "Technological specialization diploma holders", 
                    "Change of institution/course", "Short cycle diploma holders", 
                    "Change of institution/course (International)"
                ],
                "right": False
            },
            "Course": {
                "bins": [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991, 10000],
                "labels": [
                    "Biofuel Production Technologies", "Animation and Multimedia Design", 
                    "Social Service (evening attendance)", "Agronomy", "Communication Design", 
                    "Veterinary Nursing", "Informatics Engineering", "Equinculture", "Management", 
                    "Social Service", "Tourism", "Nursing", "Oral Hygiene", 
                    "Advertising and Marketing Management", "Journalism and Communication", 
                    "Basic Education", "Management (evening attendance)"
                ],
                "right": False
            },
            "Previous_qualification": {
                "bins": [0, 6, 15, 39, 45],
                "labels": ["Senior High School", "Associate Degree", "Bachelor Degree", "Master Degree"],
                "right": True
            },
            "Nacionality": {
                "bins": [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109, 110],
                "labels": [
                    "Portuguese", "German", "Spanish", "Italian", "Dutch", "English", "Lithuanian", 
                    "Angolan", "Cape Verdean", "Guinean", "Mozambican", "Santomean", "Turkish", 
                    "Brazilian", "Romanian", "Moldova (Republic of)", "Mexican", "Ukrainian", 
                    "Russian", "Cuban", "Colombian"
                ],
                "right": False
            },
            "Mothers_qualification": {
                "bins": [0, 6, 15, 39, 45],
                "labels": ["Senior High School", "Associate Degree", "Bachelor Degree", "Master Degree"],
                "right": True
            },
            "Fathers_qualification": {
                "bins": [0, 6, 15, 39, 45],
                "labels": ["Senior High School", "Associate Degree", "Bachelor Degree", "Master Degree"],
                "right": True
            },
            "Mothers_occupation": {
                "bins": [0, 11, 91, 131, 200],
                "labels": ["Homemaker", "Blue Collar", "White Collar", "Professional"],
                "right": False
            },
            "Fathers_occupation": {
                "bins": [0, 11, 91, 131, 200],
                "labels": ["Professional", "White Collar", "Blue Collar", "Unemployed"],
                "right": False
            },
            "Age_at_enrollment": {
                "bins": [16, 26, 36, 46, 55, 70],
                "labels": ["17-25", "26-35", "36-45", "46-55", "55+"],
                "right": True
            }
        }

        mapping_config = {
            "Daytime_evening_attendance": {1: "DayTime", 0: "Evening"},
            "Gender": {1: "Male", 0: "Female"}
        }

        # 1. Mapping (Daytime & Gender)
        for col, mapping in mapping_config.items():
            if col in categorical_features:
                data[col] = data[col].map(mapping)
        
        # 2. Bins Marital Status
        if "Marital_status" in categorical_features:
            marital_status = [
                (data["Marital_status"] == 1),
                (data["Marital_status"].isin([2, 5])),
                (data["Marital_status"].isin([3, 4, 6, 7])),
            ]
            marital_labels = ["Single", "Married", "Separated/Widowed"]
            data["Marital_status"] = np.select(
                marital_status, marital_labels, default="Unknown"
            )

        # 3. Bins Multi-columns via pd.cut
        for col, config in binning_config.items():
            if col in categorical_features:
                data[col] = pd.cut(
                    data[col], 
                    bins=config["bins"], 
                    labels=config["labels"], 
                    right=config["right"]
                )
        print("Categorical features have been binned.")
        return data

    except Exception as e:
        print(f"An error occurred while binning numerical features: {e}")


def one_hot_categorical(data, categorical_features):
    """Function to One-Hot Encoding on categorical features within the dataset."""
    try:
        print("\n[5] ONE-HOT ENCODING CATEGORICAL FEATURES")
        print("-" * 100)
        
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit and transform
        encode_arr = encoder.fit_transform(data[categorical_features])

        # Generate new column names
        encode_cols = encoder.get_feature_names_out(categorical_features)

        # Convert to DataFrame
        encode_categorical = pd.DataFrame(encode_arr, columns=encode_cols, index=data.index)

        # Drop categorical_features
        data = data.drop(columns=categorical_features)

        # Concatenate new categorical features
        data = pd.concat([data, encode_categorical], axis=1)

        print('Categorical features have been OneHotEncoded.')
        
        return data

    except Exception as e:
        [print(f"An error occurred while one hot encoder categorical features: {e}")]
# ================================================================================================
"""END OF PREPROCESSING DATA"""
# ================================================================================================


def main():
    """Main function to execute the data loading and preprocessing pipeline sequentially."""
    # 1. Load Data
    print("\n" + "=" * 100)
    print("        STARTING LOADING DATA       ")
    print("=" * 100)
    
    all_data = load_data()
    
    print("\n" + "=" * 100)
    print("     DATA LOADING COMPLETED      ")
    print("\n" + "=" * 100)

    # 2. Split all_data into numerical_features and categorical_features
   
    num_features = get_numerical_features(all_data)
    cat_features = get_categorical_features(all_data)

    # 3. Process Clean Outliers on Numerical Columns
    print("\n" + "=" * 100)
    print("        STARTING PREPROCESSING DATA       ")
    print("=" * 100)

    clean_data = clean_outlier(all_data, numerical_features=num_features)

    # 4. Standarize numerical features
    standarize_num = standardize_numerical_features(clean_data, numerical_features=num_features)

    # 5. Numerically Encode Target Label 'Status'
    col_status = encode_status(standarize_num)

    # 6. Bining categorical features
    bin_cat = binning_categorical_features(col_status, categorical_features=cat_features)

     # 7. Apply One-Hot Encoding to Transformed Categorical Features
    final_df = one_hot_categorical(bin_cat, categorical_features=cat_features)
    
    print("\n" + "=" * 100)
    print("        ALL PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY       ")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("         FINAL DATAFRAME DETAILED STRUCTURE:         ")
    print("=" * 100)
    print(final_df.info(max_cols=200))

    # 8. Save to CSV
    path_data = 'final_data_students.csv'

    if final_df is not None:
        try:
            final_df.to_csv(path_data, index=False)
            print(f'\n Data successfully saved to: {path_data}')
        
        except Exception as e:
            print(f"\n[ERROR] Failed to save data to {path_data}. Reason: {e}")



if __name__ == "__main__":
    main()
        