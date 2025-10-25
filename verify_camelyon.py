# verify_camelyon.py
import os
import h5py

def verify_camelyon_files(data_dir):
    """Verify that Camelyon HDF5 files are properly extracted"""
    print("Verifying Camelyon dataset files...")
    
    files_to_check = [
        'camelyonpatch_level_2_split_train_x.h5',
        'camelyonpatch_level_2_split_valid_x.h5', 
        'camelyonpatch_level_2_split_test_x.h5',
        'camelyonpatch_level_2_split_train_y.h5',
        'camelyonpatch_level_2_split_valid_y.h5',
        'camelyonpatch_level_2_split_test_y.h5'
    ]
    
    all_good = True
    
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Missing: {filename}")
            all_good = False
            continue
            
        # Try to open the HDF5 file
        try:
            with h5py.File(filepath, 'r') as f:
                if 'x' in f or 'y' in f:  # Check for expected datasets
                    file_size = os.path.getsize(filepath) / (1024**3)  # Size in GB
                    print(f"✅ {filename} - {file_size:.2f} GB")
                else:
                    print(f"❌ {filename} - Invalid HDF5 structure")
                    all_good = False
        except Exception as e:
            print(f"❌ {filename} - Error opening: {e}")
            all_good = False
    
    if all_good:
        print("\n🎉 All files verified successfully!")
        print("You can now run: python main.py --mode test_camelyon")
    else:
        print("\n❌ Some files are missing or corrupted!")
        
    return all_good

if __name__ == "__main__":
    data_dir = "data/raw/camelyon17"
    verify_camelyon_files(data_dir)