import os
import numpy as np
import cupy as cp
import dask.array as da

import matplotlib.pyplot as plt
import seaborn as sns

from util import claim
from util import free_vram

from time import time

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler, ShuffleSplit, cross_val_score

from sklearn.preprocessing import PowerTransformer
from dask_ml.preprocessing import PolynomialFeatures as dask_PolynomialFeatures
from cuml.preprocessing import PolynomialFeatures as cuml_PolynomialFeatures

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from cuml.ensemble import RandomForestRegressor

from cuml import TruncatedSVD as cuml_compressed_svd
from dask_ml.decomposition import IncrementalPCA as dask_pca
from cuml.decomposition import IncrementalPCA as cuml_pca

from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
from cuml.metrics.trustworthiness import trustworthiness

from cuml import TSNE


state_seed=23301522
state=np.random.RandomState(state_seed)

palette_discrete=lambda q:sns.color_palette("Set2", n_colors=q, desat=.9)
palette_contiguous=sns.color_palette("icefire", as_cmap=True)


def interact_feature(darr,n=2,backend='cupy')->tuple:
    if darr.ndim==2:
      if backend=='dask':
        interacter=dask_PolynomialFeatures(degree=n)
      elif backend=='cupy':
        interacter=cuml_PolynomialFeatures(degree=n)
      Xp=interacter.fit_transform(darr)
      return interacter,Xp
    raise ValueError(f"{darr.ndim=}")


def evaluate_imputation(x,y):
  splitter=ShuffleSplit(n_splits=10**1,random_state=state)
  transformer=PowerTransformer()
  regressor=RandomForestRegressor(max_features=.5,bootstrap=True,random_state=state)
  imputer=[SimpleImputer(),KNNImputer(),IterativeImputer(sample_posterior=True)]

  result={}
  for _imputer in imputer:
    processor=Pipeline(steps=[
      ("Transformer",transformer),
      ("Imputer",_imputer),
      ("Regressor",regressor)
    ])

    result[_imputer.__repr__()]=cross_val_score(
      processor,x,y,scoring="neg_mean_absolute_error",cv=splitter
    )

  sns.barplot(result.values(),errorbar="se",width=.5).set_title("MAE on Imputers")

  return result


def pca(
  darr,
  backend='dask',
  n=2,
  batch_size=1024,
  iterated_power=30,
  frontend=None
)->tuple:

  if backend=='dask':
    decomposer=dask_pca(
      n_components=n,
      batch_size=batch_size,
      iterated_power=iterated_power,
      random_state=state
    )
  else:
    decomposer=cuml_pca(
      n_components=n,
      batch_size=int(batch_size / 8),
      output_type=frontend
    )

  return decomposer,decomposer.fit_transform(darr)


def get_svd(
  darr,
  n=100,
  backend='cupy'
)->tuple:
  if backend=='dask':
    return da.linalg.svd_compressed(
      darr,
      k = n,
      compute = True
    )
  else:
    return cuml_compressed_svd(
      darr,
      algorithm = 'Jacobi',
      n_components = n,
      n_iter = 20
    )


def cluster_plot(
  Xr,
  label,
  figsize=(5,5),
  title='cluster_plot',
  output_path='result',
  output_file_suffix='cluster_plot',
  return_figure=True,
):

  if Xr.ndim==2:
    Xl=da.unique(label).compute()
    fig,ax=plt.subplots(figsize=figsize)
    for color,l in zip(palette_discrete(Xl.size),Xl):
      plt.scatter(
        Xr[label==l, 0],
        Xr[label==l, 1],
        color=color,
        label=str(l),
        alpha=.7
      )
    ax.set_title(title)
    ax.set_xlabel("Xr0")
    ax.set_ylabel("Xr1")
    plt.xticks([])
    plt.yticks([])
    epoch_str=str(time()).replace('.','')[:8]
    plt.savefig(
      os.path.join(
        os.getcwd(),
        output_path,
        f'{epoch_str}-{output_file_suffix}.png'
      ),
      transparent=True
    )

    if return_figure:
      return fig,ax
    else:
      plt.clf()
      return None
  
  raise ValueError(f'{Xr.ndim=}')


def cluster_iterated(
  X,
  clusterer,
  clusterer_options,
  cluster_count_range=range(3,11),
  scoring=None,
)->tuple:

  result={}
  for cluster_count in cluster_count_range:
    clusterer_options['n_clusters']=cluster_count
    claim(
	    f"Input: {X.shape}, {X.dtype}, {X.nbytes//1024**2}MB, {cluster_count=}"
    )
		
    t0=time()
    clusterer_=clusterer(**clusterer_options)
    clusterer_.fit(X)
    claim(f'Elements: {cp.unique(clusterer_.labels_,return_counts=True)[1]}')
    claim(f'Fitting Took {(time() - t0) // 1} s')

    if scoring:
      t0=time()
      score_historical=-1
      Xl=clusterer_.labels_

      if scoring=='silhouette':
        score=cython_silhouette_score(X,Xl,chunksize=1024*12)
      else:
        score=trustworthiness(X,Xl,batch_size=384)

      claim(f"Score: {score}")
      claim(f"Scoring Took {(time() - t0) // 1} s")

      if score>score_historical:
        score_historical=score

      result_intermidiate=(score,clusterer_)

    else:
      result_intermidiate=(1,clusterer_)
    
    result[cluster_count]=result_intermidiate
    
    result_intermidiate,score=(None,None)
    free_vram()

  if scoring:
    fig,ax=plt.subplots(figsize=(6,6))
    ax.plot(cluster_count_range,[q[0] for q in result.values()])
    ax.set_xlabel("n")
    ax.set_ylabel("Score")
    ax.set_title(f"Score, {cluster_count_range=}")
    plt.savefig(
      f"./result/{str(time()).replace('.','')[:10]}-cluster.png",
      transparent=True
    )
    plt.show()
    return result

  return result


def cluster_aggregated(
	X,
	label,
  clusterer_options=dict(
    perplexity=range(10,101,5),
    n_neighbors=[10**q for q in range(2,4)],
    n_iter=[10**q for q in range(2,4)],
    learning_rate=[10**q for q in range(-2,2)]
  ),
  clusterer_options_size=3,
  random_state_seed=state_seed,
):

  if isinstance(label, cp.ndarray):
    label=cp.asnumpy(label)

  clusterer_options_list=ParameterSampler(
    clusterer_options,
    n_iter=clusterer_options_size,
  )

  result_total=[]
  for param in clusterer_options_list:
    t0=time()
    claim(f"Fitting for ({param=})")

    decomposer=TSNE(
      **param,
      method='fft',
      exaggeration_iter=int(np.sqrt(param["n_iter"])),
      random_state=random_state_seed
    )

    result=decomposer.fit_transform(X)

    result=cp.asnumpy(result)

    cluster_plot(
      Xr=result,
      label=label,
      title=str(param),
      output_file_suffix='cluster_aggregated'
    )

    result_total.append((param,result))
    claim(f"Took {(time() - t0) // 1} s")

  free_vram()

  return result_total

