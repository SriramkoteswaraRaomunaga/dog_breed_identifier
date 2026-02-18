import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_dataset():
    """
    Organize dataset into train/test splits based on labels.csv
    """
    # Read labels CSV
    labels_csv_path = Path('dataset/labels.csv')
    if not labels_csv_path.exists():
        print(f"Error: {labels_csv_path} not found.")
        return

    labels_df = pd.read_csv(labels_csv_path)
    
    # Create main directories
    base_dir = Path('dataset')
    train_dir = base_dir / 'train'
    test_dir = base_dir / 'test'
    
    for directory in [train_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Get unique breeds
    breeds = labels_df['breed'].unique()
    
    # Create breed subdirectories in train and test
    for breed in breeds:
        (train_dir / breed).mkdir(exist_ok=True)
        (test_dir / breed).mkdir(exist_ok=True)
    
    # Split data
    train_df, test_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['breed'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")

    def copy_images(df, destination_root):
        count = 0
        missing = 0
        for _, row in df.iterrows():
            image_id = row['image_id']
            breed = row['breed']
            
            # Source path (assuming images are in dataset/images)
            src_path = base_dir / 'images' / image_id
            
            # Destination path - use basename to avoid redundant subfolders if image_id has them
            dst_path = destination_root / breed / Path(image_id).name
            
            # Copy file if exists
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                count += 1
            else:
                missing += 1
                # print(f"Warning: {image_id} not found in dataset/images/")
        return count, missing

    # Copy training images
    print("Copying training images...")
    train_count, train_missing = copy_images(train_df, train_dir)
    print(f"Copied {train_count} training images. Missing: {train_missing}")

    # Copy testing images
    print("Copying testing images...")
    test_count, test_missing = copy_images(test_df, test_dir)
    print(f"Copied {test_count} testing images. Missing: {test_missing}")
    
    print("\nDataset organization complete!")
    if train_missing > 0 or test_missing > 0:
        print("Note: Some images were missing. Please ensure 'dataset/images' contains the source images.")

if __name__ == "__main__":
    organize_dataset()
