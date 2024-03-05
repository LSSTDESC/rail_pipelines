import h5py
import copy
import numpy as np

def copy_and_duplicate_n_times(file, copy_field, n):
    print(f"copying {copy_field} field:")
    if copy_field == "data":
        inner_name = "yvals"
    elif copy_field == "ancil":
        inner_name = "zmode"
    else:
        return
    
    copy_from = f"{copy_field}_old"
    copy_to = copy_field

    data = file[copy_from]

    # remove existing data
    if copy_to in file:
        del file[copy_to]
    
    # create new data
    new_data = []
    counter = 0
    for item in data[inner_name]:
        print(f"copying original row {counter} ({n} times)...")
        for i in range(n):
            new_item = np.copy(item)
            random_values = np.random.uniform(0.995, 1.005, size=new_item.shape)
            new_item *= random_values
            new_data.append(new_item)
        counter += 1
    file.create_dataset(f"{copy_to}/{inner_name}", data=new_data)
    
    print(f"done with {copy_field}")
    

def temp_edit_data():
    print("editing data...")
    with h5py.File('short-output_BPZ_lite.hdf5', 'r+') as file:
        #n = 1 #500_000
        #copy_and_duplicate_n_times(file, "data", n)
        #copy_and_duplicate_n_times(file, "ancil", n)
        del file["ancil_old"]
        del file["data_old"]
        print("done.")


if __name__ == "__main__":
    temp_edit_data()
