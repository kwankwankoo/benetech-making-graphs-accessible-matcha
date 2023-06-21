import pandas as pd

# Read CSV file into DataFrame
data = pd.read_csv('./datasets/train_with_df_large_cwq.csv')

# Create an empty list to store the indices of lines with empty fields
lines_with_empty_fields = []

# Loop over each line in the DataFrame
for index, row in data.iterrows():
    # Check if any field in the line is empty
    if row.isnull().values.any():
        # Print the line and its content
        print("Line:", index)
        print("Content:", row)
        print()
        # Add the index to the list of lines with empty fields
        lines_with_empty_fields.append(index)

print(f"{len(lines_with_empty_fields)} empty lines found")

# Remove lines with empty fields from the DataFrame
data = data.drop(lines_with_empty_fields)

# Write the updated DataFrame to a new CSV file
data.to_csv('./datasets/train_with_df_large_cwq2.csv', index=False)