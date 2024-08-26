import os
import pandas as pd

def delete_files_based_on_status(excel_path, directory):
    # Load the Excel file
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    df = pd.read_excel(excel_path)

    # Check if the required columns are present
    if 'Filename' not in df.columns or 'Status' not in df.columns:
        raise ValueError("The Excel file must contain 'Filename' and 'Status' columns.")

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        filename = row['Filename']
        status = row['Status']

        # Only delete files with a status of 'No'
        if status == 'No':
            file_path = os.path.join(directory, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            else:
                print(f"File not found, could not delete: {file_path}")

if __name__ == "__main__":
    
    # Path to the Excel file
    excel_path = 'data_preprocessing/train_image_pair_status.xlsx'  
    
    # Directory containing the files
    directory = 'datasets/BCI/train'  

    # Run the delete function
    delete_files_based_on_status(excel_path, directory)
