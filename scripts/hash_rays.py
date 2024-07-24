import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict
import random
from multiprocessing import Process, Queue , Array

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
      return self.entry[0] == other.entry[0] and self.entry[1] == other.entry[1]
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

class HashedRay:
  def __init__(self, hash, tid: int, rayid: int):
    self.hash = hash
    self.tid = tid # thread id, index to the thread_list
    self.rayid = rayid # ray id, index to the entry in the thread_list

  def __str__(self):
    return f"HashedRay: hash:{self.hash}, tid:{self.tid}, rayid:{self.rayid}"

# Quantize direction to a sphere - xyz to theta and phi
# `theta_bits` is used for theta, `theta_bits` + 1 is used for phi, for a total of
# 2 * `theta_bits` + 1 bits
def hash_direction_spherical(d, num_sphere_bits):
  theta_bits = np.uint32(num_sphere_bits)
  phi_bits = np.uint32(theta_bits + 1)

  theta = np.uint64(np.arccos(np.clip(d[2], -1.0, 1.0)) / np.pi * 180)
  phi = np.uint64((np.arctan2(d[1], d[0]) + np.pi) / np.pi * 180)
  q_theta = theta >> np.uint64(8 - theta_bits)
  q_phi = phi >> np.uint64(9 - phi_bits)

  return (q_phi << theta_bits) | q_theta

def hash_origin_grid(o, min_val, max_val, num_bits):
  grid_size = 1 << num_bits

  hash_o_x = np.uint64(np.clip((o[0] - min_val[0]) / (max_val[0] - min_val[0]) * grid_size, 0.0, float(grid_size) - 1))
  hash_o_y = np.uint64(np.clip((o[1] - min_val[1]) / (max_val[1] - min_val[1]) * grid_size, 0.0, float(grid_size) - 1))
  hash_o_z = np.uint64(np.clip((o[2] - min_val[2]) / (max_val[2] - min_val[2]) * grid_size, 0.0, float(grid_size) - 1))
  
  hash_value = (hash_o_x << np.uint32((2 * num_bits))) | (hash_o_y << np.uint32(num_bits)) | hash_o_z
  return np.uint64(hash_value)

def hash_grid_spherical(ray_direction, ray_origin, min_val, max_val, num_grid_bits, num_sphere_bits):
  hash_d = hash_direction_spherical(ray_direction, num_sphere_bits)
  hash_o = hash_origin_grid(ray_origin, min_val, max_val, num_grid_bits)
  hash_value = hash_o ^ hash_d

  return hash_value

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

def load_bunny():
  print("Loading bunny")
  # Read the first CSV file into a DataFrame
  df1 = pd.read_csv('../../outputs/bunny_left_eye_mem_access.csv')
  # Convert the DataFrame to a list of lists
  csv1 = df1.values.tolist()
  number_of_threads1 = df1['tid'].nunique()

  thread_list1 = parse_csv(csv1,number_of_threads1)

  # Read the second CSV file into a DataFrame
  df2 = pd.read_csv('../../outputs/bunny_right_eye_mem_access.csv')
  # Convert the DataFrame to a list of lists
  csv2 = df2.values.tolist()
  number_of_threads2 = df2['tid'].nunique()

  thread_list2 = parse_csv(csv2,number_of_threads2)

  min_val = np.array([0.0, 0.0, -555.0])  # bunny min values
  max_val = np.array([556.0, 556.0, 1.0])  # bunny max values

  return thread_list1,thread_list2,min_val,max_val

def load_sponza():
  print("Loading sponza")
  # Read the first CSV file into a DataFrame
  df1 = pd.read_csv('../../outputs/sponza_left_eye_mem_access.csv')
  # Convert the DataFrame to a list of lists
  csv1 = df1.values.tolist()
  number_of_threads1 = df1['tid'].nunique()

  thread_list1 = parse_csv(csv1,number_of_threads1)

  # Read the second CSV file into a DataFrame
  df2 = pd.read_csv('../../outputs/sponza_right_eye_mem_access.csv')
  # Convert the DataFrame to a list of lists
  csv2 = df2.values.tolist()
  number_of_threads2 = df2['tid'].nunique()

  thread_list2 = parse_csv(csv2,number_of_threads2)

  min_val = np.array([-1105.42603,-126.442497,-1920.94592]) # sponza min values
  max_val = np.array([1198.57397,1433.5575,1807.05408]) # sponza max values

  return thread_list1,thread_list2,min_val,max_val

def hash_rays(thread_list1, thread_list2, min_val, max_val, num_grid_bits, num_sphere_bits):
  print("Hashing rays...")
  hash_list1 = []
  for tid,thread in enumerate(thread_list1):
    for rayid,ray in enumerate(thread):
      # only hash if the ray intersects the scene
      if(any(obj.entry[2] == 5 for obj in ray.mem_entries)):
        hash_list1.append(HashedRay(hash_grid_spherical(np.array([ray.ray_data.ray_dir_x, ray.ray_data.ray_dir_y, ray.ray_data.ray_dir_z]),
                                            np.array([ray.ray_data.ray_orig_x, ray.ray_data.ray_orig_y, ray.ray_data.ray_orig_z]),
                                            min_val, max_val,num_grid_bits,num_sphere_bits), tid, rayid))
      
  hash_list2 = []
  for tid,thread in enumerate(thread_list2):
    for rayid,ray in enumerate(thread):
      if(any(obj.entry[2] == 5 for obj in ray.mem_entries)):
        hash_list2.append(HashedRay(hash_grid_spherical(np.array([ray.ray_data.ray_dir_x, ray.ray_data.ray_dir_y, ray.ray_data.ray_dir_z]),
                                            np.array([ray.ray_data.ray_orig_x, ray.ray_data.ray_orig_y, ray.ray_data.ray_orig_z]),
                                            min_val, max_val,num_grid_bits,num_sphere_bits), tid, rayid))

  hash_list1.sort(key=lambda x:x.hash)
  hash_list2.sort(key=lambda x:x.hash)
  return hash_list1,hash_list2

def hash_rays_random(thread_list1, thread_list2):
  print("Hashing rays randomly...")
  hash_list1 = []
  for tid,thread in enumerate(thread_list1):
    for rayid,ray in enumerate(thread):
      # only hash if the ray intersects the scene
      if(any(obj.entry[2] == 5 for obj in ray.mem_entries)):
        hash_list1.append(HashedRay(random.randint(0, 32678), tid, rayid))
      
  hash_list2 = []
  for tid,thread in enumerate(thread_list2):
    for rayid,ray in enumerate(thread):
      if(any(obj.entry[2] == 5 for obj in ray.mem_entries)):
        hash_list2.append(HashedRay(random.randint(0, 32678) ,tid, rayid))

  return hash_list1,hash_list2

def check_matches(list1: List[MemEntry], list2: List[MemEntry]):
  num_matches = 0
  length = len(list2) if len(list2) < len(list1) else len(list1)
  for idx in range(length):
      if(list1[idx].entry[0] == list2[idx].entry[0] and list1[idx].entry[1] == list2[idx].entry[1]):
        num_matches += 1
      else:
        break
  
  return num_matches

# Find all matching hashes, counts the matching memory entries and averages them
def find_all_matches_best(hash_list1, hash_list2, thread_list1, thread_list2, start, length, full_match_list,full_len_list,full_tid_list,full_rayid_list):
  if start+length > len(hash_list2):
    length = len(hash_list2) - start  
  for idx,ray2 in enumerate(hash_list2[start:start+length]):
    found = 0
    num_matching_hashes = 1
    total_matches = 0
    mem_len = len(thread_list2[ray2.tid][ray2.rayid].mem_entries)
    for ray1 in hash_list1:
      if(ray1.hash == ray2.hash):
        matches = check_matches(thread_list1[ray1.tid][ray1.rayid].mem_entries, thread_list2[ray2.tid][ray2.rayid].mem_entries)
        num_matching_hashes += 1
        total_matches += matches
        #best_tid = ray1.tid
        #best_rayid = ray1.rayid
        found = 1
      elif found:
        break
    if num_matching_hashes > 1:
      num_matching_hashes -= 1
    full_match_list[start+idx] = np.int32(np.ceil(total_matches/num_matching_hashes))
    full_len_list[start+idx] = mem_len
    #full_tid_list[idx] = best_tid
    #full_rayid_list[idx] = best_rayid
    # print(f"hash: {str(ray2.hash).ljust(5)}, {str(best_matches).rjust(4)}/{str(best_len).ljust(4)}, "
    #       f"tid1,rayid1: {str(best_tid).rjust(5)},{str(best_rayid).rjust(2)}, "
    #       f"tid2,rayid2: {str(ray2.tid).rjust(5)},{str(ray2.rayid).rjust(2)}, "
    #       f"raydata1: {thread_list1[best_tid][best_rayid].ray_data}, " 
    #       f"raydata2: {thread_list2[ray2.tid][ray2.rayid].ray_data}")

def find_all_matches_best_parallel(hash_list1, hash_list2, thread_list1, thread_list2, n=10):
  full_length = len(hash_list2)
  length = np.int32(np.ceil(full_length / n))

  full_match_list = Array('i',full_length)
  full_len_list = Array('i',full_length)
  full_tid_list = Array('i',full_length)
  full_rayid_list = Array('i',full_length)

  procs = [[] for _ in range(n)]

  for i in range(n):
    procs[i] = Process(target=find_all_matches_best, args=(hash_list1,hash_list2,thread_list1,thread_list2,i*length,length,full_match_list,full_len_list,full_tid_list,full_rayid_list))
    procs[i].start()
  for i in range(n):
    procs[i].join()

  return np.array(full_match_list), np.array(full_len_list), np.array(full_tid_list), np.array(full_rayid_list)

def find_all_matches_nearest(hash_list1, hash_list2, thread_list1, thread_list2, start, length,full_match_list,full_len_list,full_tid_list,full_rayid_list):
  if start+length > len(hash_list2):
    length = len(hash_list2) - start
  print(f"{start},{length}")
  for idx,ray2 in enumerate(hash_list2[start:start+length]):
    nearest_matches = 0
    nearest_len = len(thread_list2[ray2.tid][ray2.rayid].mem_entries)
    nearest_tid = 0
    nearest_rayid = 0
    found = 0
    min_distance = 32768
    for ray1 in hash_list1:
      if(ray1.hash == ray2.hash):
        if abs(ray1.tid-ray2.tid) < min_distance:
          min_distance = abs(ray1.tid-ray2.tid)
          matches = check_matches(thread_list1[ray1.tid][ray1.rayid].mem_entries, thread_list2[ray2.tid][ray2.rayid].mem_entries)
          nearest_matches = matches
          nearest_tid = ray1.tid
          nearest_rayid = ray1.rayid
          found = 1
      elif found:
        break
    full_match_list[start+idx] = nearest_matches
    full_len_list[start+idx] = nearest_len
    full_tid_list[start+idx] = nearest_tid
    full_rayid_list[start+idx] = nearest_rayid

def find_all_matches_nearest_parallel(hash_list1, hash_list2, thread_list1, thread_list2, n=10):
  
  full_length = len(hash_list2)
  length = np.int32(np.ceil(full_length / n))

  full_match_list = Array('i',full_length)
  full_len_list = Array('i',full_length)
  full_tid_list = Array('i',full_length)
  full_rayid_list = Array('i',full_length)

  procs = [[] for _ in range(n)]

  for i in range(n):
    procs[i] = Process(target=find_all_matches_nearest, args=(hash_list1,hash_list2,thread_list1,thread_list2,i*length,length,full_match_list,full_len_list,full_tid_list,full_rayid_list))
    procs[i].start()
  for i in range(n):
    procs[i].join()

  return np.array(full_match_list), np.array(full_len_list), np.array(full_tid_list), np.array(full_rayid_list)

# # get rid of multiple entries by picking the entry
# def compress_hash_list_last(hash_list):
#   # sort by hashes first
#   hash_list.sort(key=lambda x:x.hash)
