
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class Indices:
  IDX = 0
  TID = 1
  ADDR = 2
  SIZE = 3
  TYPE = 4
  RAY_ORIG_X = 5
  RAY_ORIG_Y = 6
  RAY_ORIG_Z = 7
  RAY_DIR_X = 8
  RAY_DIR_Y = 9
  RAY_DIR_Z = 10

class RayData:
  def __init__(self, tid, ray_orig_x, ray_orig_y, ray_orig_z, ray_dir_x, ray_dir_y, ray_dir_z):
    self.tid = tid

    self.ray_orig_x=ray_orig_x
    self.ray_orig_y=ray_orig_y
    self.ray_orig_z=ray_orig_z

    self.ray_dir_x=ray_dir_x
    self.ray_dir_y=ray_dir_y
    self.ray_dir_z=ray_dir_z

  def __eq__(self, other):
    if isinstance(other, RayData):
      # Compare attributes for equality
      return  self.tid == other.tid and \
              self.ray_orig_x == other.ray_orig_x and \
              self.ray_orig_y == other.ray_orig_y and \
              self.ray_orig_z == other.ray_orig_z and \
              self.ray_dir_x == other.ray_dir_x and \
              self.ray_dir_y == other.ray_dir_y and \
              self.ray_dir_z == other.ray_dir_z
    return False
  
  def __str__(self):
    # Customize the string representation for printing
    return f"ray_data instance: {self.tid}, {self.ray_orig_x}, {self.ray_orig_y}, {self.ray_orig_z}, {self.ray_dir_x}, {self.ray_dir_y}, {self.ray_dir_z}"

class MemEntry:
  def __init__(self, address, size, type):
    self.entry = [address, size, type]

  def __eq__(self, other):
    if isinstance(other, MemEntry):
      # Compare attributes for equality
      return self.entry == other.entry
    return False
  
  def __str__(self):
    # Customize the string representation for printing
    return f"mem_entry instance: {self.entry}"

class TraceRayEntry:
  def __init__(self, ray_data: RayData, mem_entries: List[MemEntry]):
    self.ray_data = ray_data
    self.mem_entries = mem_entries

  def __str__(self):
    # Customize the string representation for printing
    return f"trace_ray_entry instance: {self.ray_data}, {self.mem_entries}"

def read_csv_to_list(file_path):
  # Read the CSV file into a DataFrame
  df = pd.read_csv(file_path)
  
  # Convert the DataFrame to a list of lists
  data_list = df.values.tolist()
  
  return data_list

def compare_mem_entries(entry1: List[MemEntry], entry2: List[MemEntry]):
  return entry1 == entry2

def check_similarity():
  # Check if two file paths are provided as command-line arguments
  if len(sys.argv) != 3:
    print("Usage: python script.py file1.csv file2.csv")
    sys.exit(1)

  # Get file paths from command-line arguments
  file1_path = sys.argv[1]
  file2_path = sys.argv[2]

  # Read CSV files into lists of lists
  csv1 = read_csv_to_list(file1_path)
  csv2 = read_csv_to_list(file2_path)

  left_eye_entries = []
  mem_entries = []
  last_tidx = csv1[0][0]
  last_ray_data = RayData(csv1[0][4],csv1[0][5],csv1[0][6],csv1[0][7],csv1[0][8],csv1[0][9])
  print(last_tidx)
  print(last_ray_data)
  for entry in csv1:
    if(last_tidx != entry[0]):
      left_eye_entries.append(TraceRayEntry(last_ray_data,mem_entries[:]))
      mem_entries.clear()
      last_ray_data = RayData(entry[4],entry[5],entry[6],entry[7],entry[8],entry[9])
      last_tidx = entry[0]
    
    mem_entries.append(MemEntry(entry[1],entry[2],entry[3]))

  print("Finished parsing left eye csv...")
  last_tidx = csv2[0][0]
  last_ray_data = RayData(csv2[0][4],csv2[0][5],csv2[0][6],csv2[0][7],csv2[0][8],csv2[0][9])
  mem_entries.clear()
  rt_idx = 0
  total_entries = 0
  similar_entries = 0
  for entry in csv2:
    if(last_tidx != entry[0]):
      if left_eye_entries[rt_idx].mem_entries == mem_entries:
        similar_entries += 1
      rt_idx = rt_idx if len(left_eye_entries)-1 == rt_idx else rt_idx+1
      mem_entries.clear()
      last_ray_data = RayData(entry[4],entry[5],entry[6],entry[7],entry[8],entry[9])
      last_tidx = entry[0]
      total_entries += 1

    mem_entries.append(MemEntry(entry[1],entry[2],entry[3]))
  print(f"total entries: {total_entries}")
  print(f"similar entries: {similar_entries}")

def parse_csv(csv, number_of_threads):
  thread_list = [[] for _ in range(number_of_threads)]
  mem_entries = []
  last_idx = csv[0][Indices.IDX]
  last_tid = csv[0][Indices.TID]
  last_ray_data = RayData(csv[0][Indices.TID],
                           csv[0][Indices.RAY_ORIG_X],
                           csv[0][Indices.RAY_ORIG_Y],
                           csv[0][Indices.RAY_ORIG_Z],
                           csv[0][Indices.RAY_DIR_X],
                           csv[0][Indices.RAY_DIR_Y],
                           csv[0][Indices.RAY_DIR_Z])
  print("Parsing the csv...")
  for entry in csv:
    if(last_idx != entry[Indices.IDX]):
      thread_list[last_tid].append(TraceRayEntry(last_ray_data,mem_entries[:]))
      mem_entries.clear()
      last_ray_data = RayData(entry[Indices.TID],
                               entry[Indices.RAY_ORIG_X],
                               entry[Indices.RAY_ORIG_Y],
                               entry[Indices.RAY_ORIG_Z],
                               entry[Indices.RAY_DIR_X],
                               entry[Indices.RAY_DIR_Y],
                               entry[Indices.RAY_DIR_Z])
      last_idx = entry[Indices.IDX]
      last_tid = entry[Indices.TID]
    
    mem_entries.append(MemEntry(entry[Indices.ADDR],entry[Indices.SIZE],entry[Indices.TYPE]))
  print("Finished parsing the csv...")
  return thread_list

def generate_numofrays_heatmap_matrix(thread_list, number_of_threads):
  print("Generating number of rays heatmap...")
  image_dim = int(np.sqrt(number_of_threads))
  heatmap = np.zeros((image_dim,image_dim))
  for y in range(image_dim):
    for x in range(image_dim):
      heatmap[y,x] = len(thread_list[x+y*image_dim])

  print(f"Max number of rays traced: {np.max(heatmap)}")
  return heatmap

def generate_numofmemacc_heatmap_matrix(thread_list, number_of_threads):
  print("Generating number of memory accesses heatmap...")
  image_dim = int(np.sqrt(number_of_threads))
  heatmap = np.zeros((image_dim,image_dim))
  for y in range(image_dim):
    for x in range(image_dim):
      for entry in thread_list[x+y*image_dim]:
        heatmap[y,x] += len(entry.mem_entries)

  print(f"Max number of memory accesses: {np.max(heatmap)}")
  return heatmap

if __name__ == "__main__":
  if len(sys.argv) != 5:
    print("Usage: python script.py file1.csv file2.csv file3.csv file4.csv")
    sys.exit(1)

  # Get file paths from command-line arguments
  file_paths = sys.argv[1:]

  fig, axs = plt.subplots(2, 2, figsize=(10, 10))

  for i in range(2):
      for j in range(2):
          # Read the CSV file into a DataFrame
          df = pd.read_csv(file_paths[i * 2 + j])
          # Convert the DataFrame to a list of lists
          csv = df.values.tolist()
          number_of_threads = df['tid'].nunique()

          thread_list = parse_csv(csv, number_of_threads)
          heatmap = generate_numofmemacc_heatmap_matrix(thread_list, number_of_threads)

          im = axs[i, j].imshow(heatmap, cmap='hot', interpolation='nearest')
          axs[i, j].set_title(f'Heatmap {i * 2 + j + 1}')
          fig.colorbar(im, ax=axs[i, j])

  plt.show()




  # # Get file paths from command-line arguments
  # file_path1 = sys.argv[1]
  # file_path2 = sys.argv[2]

  # # Read the first CSV file into a DataFrame
  # df1 = pd.read_csv(file_path1)
  # # Convert the DataFrame to a list of lists
  # csv1 = df1.values.tolist()
  # number_of_threads1 = df1['tid'].nunique()

  # thread_list1 = parse_csv(csv1,number_of_threads1)
  # heatmap1 = generate_numofmemacc_heatmap_matrix(thread_list1, number_of_threads1)

  # # Read the second CSV file into a DataFrame
  # df2 = pd.read_csv(file_path2)
  # # Convert the DataFrame to a list of lists
  # csv2 = df2.values.tolist()
  # number_of_threads2 = df2['tid'].nunique()

  # thread_list2 = parse_csv(csv2,number_of_threads2)
  # heatmap2 = generate_numofmemacc_heatmap_matrix(thread_list2, number_of_threads2)

  # # Display both heatmaps
  # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

  # im1 = axs[0].imshow(heatmap1, cmap='hot', interpolation='nearest')
  # axs[0].set_title('Heatmap 1')
  # fig.colorbar(im1, ax=axs[0])

  # im2 = axs[1].imshow(heatmap2, cmap='hot', interpolation='nearest')
  # axs[1].set_title('Heatmap 2')
  # fig.colorbar(im2, ax=axs[1])

  # plt.show()