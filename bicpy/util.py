import os
import numpy as np
import cupy as cp
import dask.array as da
import zarr
import rmm

from rmm.allocators.cupy import rmm_cupy_allocator
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

from shutil import make_archive
from time import time


memory_pool=(cp.get_default_memory_pool(),cp.get_default_pinned_memory_pool())
data_order=("arr","ind","col")


def free_vram(memory_pool):
  for pool in memory_pool:
    pool.free_all_blocks()
  return 


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


def get_rmm_client(n_thread=4,port=15220):
  cluster=LocalCUDACluster(
    scheduler_port=port,
    threads_per_worker=n_thread,
    local_directory=f'/tmp/rmm-temp-{str(time())[:-8]}',
    rmm_log_directory=f'/tmp/rmm-log-{str(time())[:-8]}',
    rmm_managed_memory=True
  )

  client=Client(cluster)

  client.run(cp.cuda.set_allocator,rmm_cupy_allocator)
  rmm.reinitialize(managed_memory=True)
  cp.cuda.set_allocator(rmm_cupy_allocator)

  return client


def is_sound(data:tuple)->bool:
  result=True if len(data_order)==len(data) else False
  return result


def is_darr(arr)->bool:
  return True if isinstance(arr, da.core.Array) else False


def to_darr(data, blocksize='auto', thin=False)->da.core.Array:
  axis = np.argmax(data.shape)
  if is_darr(data) and thin:
    darr = da.rechunk(
      data,
      chunks = {axis: data.shape[axis]}
    )
  else:
    if thin and data.ndim == 2:
      darr = da.rechunk(
        da.from_array(data, chunks = blocksize),
        chunks = {axis: data.shape[axis]}
      )
    else:
        darr = da.from_array(
          data,
          chunks = blocksize
        )
  return darr


def to_darr_all(data:tuple)->tuple:
  if is_sound(data):
    return to_darr(data[0],"32MB",thin=True),to_da(data[1]),to_da(data[2])


def parse_data(obj:dict,dry_run=True)->tuple:
  if dry_run:
    arr=np.asarray([*obj["data"]["arr"][:50],*obj["data"]["arr"][-50:]])
  else:
    arr=np.asarray(obj["data"]["arr"])
  ind=np.asarray(obj["data"]["ind"],dtype="U32") if \
    "ind" in obj["data"] and obj["data"]["ind"] else np.arange(arr.shape[0])
  col=np.asarray(obj["data"]["col"],dtype="U32") if \
    "col" in obj["data"] and obj["data"]["col"] else np.arange(arr.shape[1])
  return arr,ind,col


def init_zarr(root_path,data_name)->zarr.Group:
  storage=zarr.storage.DirectoryStore(os.path.join(root_path,data_name))
  return zarr.open(store=storage,mode="r+")


def write_init(root_path,data_name,data)->zarr.Group:
  group=init_zarr(root_path,data_name)
  if len(data)==len(data_order):
    for dataset in zip(data_order,data):
      group.create_dataset(name=dataset[0],data=dataset[1].compute())
    print(group.info)
    return group
  raise ValueError(f'Data seems to unlikely have elements of {data_order}')


def write_array(group:zarr.Group,*args)->list:
  if len(args[-1])==2:
    result=[]
    for name,data in args:
      if isinstance(data,da.core.Array):
        raise TypeError(f'Only computed result is supported, got {type(data)}')
      
      group.create_dataset(name=name,data=data)
      result.append((name,data.shape))

      claim(f"Created {name=} of {data.shape}, {data.dtype}")
    return result


def to_tar(root_path,archive_name)->str:
  base_name=os.path.join(root_path,archive_name)
  full_path=root_path + make_archive(base_name,"tar",root_path,archive_name)
  return full_path

