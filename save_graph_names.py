import random

def read_first_column(file_path):
    """Reads the first column from a TSV file."""
    with open(file_path, 'r') as file:
        return [line.split('\t')[0] for line in file]

def merge_columns_and_save(train_tsv_path, val_tsv_path, test_txt_path, output_file_path):
    """Reads the first column from each of two TSV files, merges them, appends contents from another text file, and shuffles the rows."""
    # Read the first column from each TSV file
    column_1 = read_first_column(train_tsv_path)
    column_2 = read_first_column(val_tsv_path)
    
    # Merge the two columns
    merged_columns = column_1 + column_2  # This appends the second list to the first
    
    # Append the contents of the test text file to the merged list
    with open(test_txt_path, 'r') as second_file:
        for line in second_file:
            merged_columns.append(line.strip())  # Use strip() to remove any trailing newline characters

    # Shuffle the merged list
    random.shuffle(merged_columns)
    
    # Save the shuffled lines to the output file
    with open(output_file_path, 'w') as output_file:
        for item in merged_columns:
            output_file.write(f"{item}\n")

    print(f"Shuffled contents saved to {output_file_path}")

# Define the paths to your files
train_file_path = './data/train.tsv'
val_file_path = './data/val.tsv'
test_txt_path = './data/test_cids.txt'
output_file_path = './data/cids.txt'

# Call the function to merge columns and save
merge_columns_and_save(train_file_path, val_file_path, test_txt_path, output_file_path)
