import numpy as np
import pandas as pd
import re
import io

def read_arff(file_path):
    """Reads an ARFF file and returns the attribute names, types, and data."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    attributes = []
    data_start = None

    # Extract attributes and locate the start of data
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith("@attribute"):
            parts = re.split(r'\s+', line, maxsplit=2)
            attributes.append((parts[1], parts[2]))  # (name, type)
        elif line.lower().startswith("@data"):
            data_start = i + 1
            break

    # Read data into a DataFrame
    data_lines = lines[data_start:]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
    df.columns = [attr[0] for attr in attributes]

    return df, attributes, lines[:data_start]

def filter_extreme_values(df, attributes, num_remove=3):
    """Removes rows containing the num_remove smallest and largest values in any numeric column."""
    numeric_columns = [attr[0] for attr in attributes if attr[1].lower() == 'numeric']
    mask = np.ones(len(df), dtype=bool)  # Start with all rows included

    for col in numeric_columns:
        sorted_indices = np.argsort(df[col])  # Get sorted indices
        mask[sorted_indices[:num_remove]] = False  # Remove smallest
        mask[sorted_indices[-num_remove:]] = False  # Remove largest

    return df[mask]

def write_arff(file_path, df, attributes, header_lines):
    """Writes a filtered DataFrame back to an ARFF file."""
    with open(file_path, 'w') as f:
        f.writelines(header_lines)  # Write attribute section
        f.write("\n@DATA\n")  # Write data section
        df.to_csv(f, index=False, header=False)

def main(input_file, output_file):
    df, attributes, header_lines = read_arff(input_file)
    filtered_df = filter_extreme_values(df, attributes)
    write_arff(output_file, filtered_df, attributes, header_lines)
    print(f"Filtered ARFF file saved as: {output_file}")

# Example usage:
input_file = "EEG_Eye_State.arff"
output_file = "filtered_output.arff"
main(input_file, output_file)