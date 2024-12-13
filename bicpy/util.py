import os
import numpy as np
import cupy as cp
import dask.array as da
import zarr
import rmm

from rmm.allocators.cupy import rmm_cupy_allocator
from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait

from shutil import make_archive
from time import time


memory_pool=(cp.get_default_memory_pool(),cp.get_default_pinned_memory_pool())
data_order=("arr","ind","col")


def free_vram(memory_pool):
  for pool in memory_pool:
    pool.free_all_blocks()
  return 


def init_rmm_client(n_thread=4)->Client:
  cluster=LocalCUDACluster(
    threads_per_worker=n_thread,
    rmm_log_directory=f'/tmp/rmm-log-{str(time())[:-8]}',
    rmm_managed_memory=True
  )

  client=Client(cluster)

  client.run(cp.cuda.set_allocator,rmm_cupy_allocator)
  rmm.reinitialize(managed_memory=True)
  cp.cuda.set_allocator(rmm_cupy_allocator)

  return client


def claim(about,what=""):
  padder="ㅡ"*5
  print(padder,about,what)
  return 


def lap(func,**kwargs):
  t0=time()
  result=func(**kwargs)
  lapsed=(time()-t0) // 1
  claim(f"{func.__repr__()} Took",f"{lapsed} s")
  return result


def is_sound(data:tuple)->bool:
  result=True if len(data_order)==len(data) else False
  return result


def to_da(data,blocksize="auto",thin=False)->da.Array:
  darr=da.from_array(data,chunks=blocksize)
  if thin:
    if darr.shape[0]>darr.shape[1] and darr.ndim==2:
      return da.rechunk(darr,chunks={1:darr.shape[1]})
    raise ValueError(f"{darr.shape[0] < darr.shape[1]}")
  return darr


def to_da_all(data:tuple)->tuple:
  if is_sound(data):
    return to_da(data[0],"32MB",thin=True),to_da(data[1]),to_da(data[2])


def parse_data(obj:dict,dry_run=True)->tuple:
  if dry_run:
    arr=np.asarray([*obj["data"]["arr"][:50],*obj["data"]["arr"][-50:]])
  else:
    arr=np.asarray(obj["data"]["arr"])
  ind=np.asarray(obj["data"]["ind"],dtype="U32") if "ind" in obj["data"] and obj["data"]["ind"] else np.arange(arr.shape[0])
  col=np.asarray(obj["data"]["col"],dtype="U32") if "col" in obj["data"] and obj["data"]["col"] else np.arange(arr.shape[1])
  return arr,ind,col


def init_zarr(root_path,data_name):
  storage=zarr.storage.DirectoryStore(os.path.join(root_path,data_name))
  return zarr.open(storage,mode="r+")


def write_init(root_path,data_name,data):
  group=init_zarr(root_path,data_name)
  for dataset in zip(data_order,data):
    group.create_dataset(name=dataset[0],data=dataset[1].compute())
  print(group.info)
  return group


def write_array(group,*args):
	if len(args[-1])==2:
		result=[]
		for name,data in args:
			group.create_dataset(name=name,data=data)
			result.append((name,data.shape))
			print(f"Created {name=} of {data.shape}, {data.dtype}")
		return result


def write_array_result(group,*args):
    if len(args[-1])==2:
      group.create_group("result")
      write_array(group["result"],args)
    return 


def to_tar(root_path,archive_name):
  base_name=os.path.join(root_path,archive_name)
  return root_path + make_archive(base_name,"tar",root_path,archive_name)
