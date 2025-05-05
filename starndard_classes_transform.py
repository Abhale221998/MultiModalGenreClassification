###common fields from image and text datasets 


# import pandas as pd

# # Load the CSV file
# data = pd.read_csv('/home/sai/Downloads/assignment/train_data_cleaned.csv')

# # View the first few rows of the CSV
# print("First few rows of the dataset:")
# print(data.head())

# # Get unique values of a specific column (e.g., 'GENRE')
# print("\nUnique values in 'GENRE' column:")
# print(data['GENRE'].unique())
# import pandas as pd

# # Load the datasets
# data1 = pd.read_csv('/home/sai/Downloads/assignment/english_with_posters_only.csv')  # Replace with your actual file name
# data2 = pd.read_csv('/home/sai/Downloads/assignment/train_data_cleaned.csv')  # Replace with your actual file name

# # Clean the 'GENRE' column by stripping any leading/trailing spaces
# data1['genre'] = data1['genre'].str.strip().str.lower()
# data2['GENRE'] = data2['GENRE'].str.strip().str.lower()

# # Get unique genres from both datasets
# unique_genres_1 = set(data1['genre'].unique())
# unique_genres_2 = set(data2['GENRE'].unique())

# # Find common genres between the two datasets
# common_genres = unique_genres_1.intersection(unique_genres_2)

# # Print the common genres
# print("Common genres between the two datasets:")
# print(common_genres)



##using the common groups preprocess the text dataset to match the Genre classes


import pandas as pd

# Load the second CSV
data2 = pd.read_csv('/home/sai/Downloads/assignment/train_data_cleaned.csv')  # Replace with your actual file name

# List of predefined genres to keep as they are
predefined_genres = {'drama', 'adventure', 'comedy', 'sci-fi', 'documentary', 'thriller', 'animation', 'horror', 'action', 'romance'}

# Clean the 'GENRE' column by stripping any leading/trailing spaces and converting to lowercase
data2['GENRE'] = data2['GENRE'].str.strip().str.lower()

# Function to convert non-predefined genres to 'others'
def convert_to_others(genre):
    if genre in predefined_genres:
        return genre
    else:
        return 'others'

# Apply the function to the 'GENRE' column
data2['GENRE'] = data2['GENRE'].apply(convert_to_others)

# Print the count of each genre
print(data2['GENRE'].value_counts())
