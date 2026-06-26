import joblib
import json
import os
import sys

# Add the current path so imports work even when the file is run from a subfolder.
sys.path.append('.')

def xem_file_pkl_joblib(duong_dan):
    print("\n" + "-"*30)
    print(f"READING FILE: {duong_dan}")

    try:
        # Load the file.
        data = joblib.load(duong_dan)
        print("Loaded successfully!")
        print(f"Data type: {type(data)}")

        # 1. If this is a model with parameters.
        # Check whether data has a parameter getter.
        if hasattr(data, 'get_params'):
            print("\n[MODEL PARAMETERS]:")
            params = data.get_params()
            print(params)

        # 2. FEATURE IMPORTANCES
        # Check whether the model exposes feature importances.
        if hasattr(data, 'feature_importances_'):
            print("\n[TOP 5 IMPORTANT FEATURES]:")

            # Get importance scores.
            scores = data.feature_importances_

            # Try to find feature names if they were saved.
            feature_names = []
            if hasattr(data, 'feature_names_in_'):
                feature_names = data.feature_names_in_
            elif hasattr(data, 'feature_names_'):
                feature_names = data.feature_names_

            # Create name/score pairs for sorting.
            danh_sach = []
            for i in range(len(scores)):
                diem = scores[i]
                # Use saved feature names when available; otherwise fall back to Feature_0, Feature_1, ...
                if len(feature_names) > i:
                    ten = feature_names[i]
                else:
                    ten = f"Feature_{i}"
                danh_sach.append((ten, diem))
            
            # Sort by descending importance score.
            danh_sach_sap_xep = sorted(danh_sach, key=lambda x: x[1], reverse=True)

            # Print the first 5 items.
            for i in range(5):
                if i < len(danh_sach_sap_xep):
                    print(f"  {i+1}. {danh_sach_sap_xep[i][0]}: {danh_sach_sap_xep[i][1]:.4f}")

        # 3. Other objects, such as a pipeline.
        # Use vars() to inspect internal attributes.
        print("\n[OTHER INFORMATION]:")
        try:
            thong_tin = vars(data)
            for ten, gia_tri in thong_tin.items():
                print(f"- {ten}: {gia_tri}")
        except:
            print("Could not read internal attribute details.")

    except Exception as loi:
        print(f"An error occurred: {loi}")

def xem_file_json(duong_dan):
    print("\n" + "-"*30)
    print(f"READING JSON FILE: {duong_dan}")

    try:
        with open(duong_dan, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("--- CONTENT ---")
        # Iterate through each JSON key.
        for key in data:
            value = data[key]

            # If the value is a long list, print only the item count.
            if type(value) == list:
                print(f"- {key}: (List with {len(value)} items)")

            # If the value is a nested dictionary, print it in more detail.
            elif type(value) == dict:
                print(f"- {key}:")
                for k_con, v_con in value.items():
                    print(f"   + {k_con}: {v_con}")

            # Otherwise, print normally.
            else:
                print(f"- {key}: {value}")

    except Exception as loi:
        print(f"JSON read error: {loi}")

def lay_danh_sach_file():
    ds_file = []
    cac_thu_muc = ['models', 'reports']

    for thu_muc in cac_thu_muc:
        # Check whether the folder exists.
        if os.path.exists(thu_muc):
            # Get all files in the folder.
            file_trong_folder = os.listdir(thu_muc)
            for ten_file in file_trong_folder:
                # Keep only the file extensions we need.
                if ten_file.endswith('.pkl') or ten_file.endswith('.joblib') or ten_file.endswith('.json'):
                    # Combine folder and file name, for example models/model.pkl.
                    duong_dan_day_du = os.path.join(thu_muc, ten_file)
                    ds_file.append(duong_dan_day_du)
    return ds_file

def main():
    while True:
        print("\nFILE INSPECTION MENU")
        
        danh_sach = lay_danh_sach_file()
        
        if len(danh_sach) == 0:
            print("No files found!")
            break

        # Print the list to the screen.
        for i in range(len(danh_sach)):
            print(f"[{i + 1}] {danh_sach[i]}")
        print("[0] Exit")

        chon = input("\nEnter the file number to inspect: ")

        if chon == '0':
            print("PROGRAM ENDED!")
            break
        
        try:
            so_thu_tu = int(chon) - 1
            
            if so_thu_tu >= 0 and so_thu_tu < len(danh_sach):
                file_duoc_chon = danh_sach[so_thu_tu]

                # Check the file extension to call the correct reader.
                if file_duoc_chon.endswith('.json'):
                    xem_file_json(file_duoc_chon)
                else:
                    xem_file_pkl_joblib(file_duoc_chon)

                input("\n(Press Enter to continue...)")
            else:
                print("Invalid number!")
        except:
            print("Please enter a number!")

if __name__ == "__main__":
    main()

    # Run the following command in the terminal:
    # python check_results.py
