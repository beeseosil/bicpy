import numpy as np
import dask.array as da
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from util import *

from time import time

from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterSampler, ShuffleSplit, cross_val_score
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from cuml.ensemble import RandomForestRegressor
from cuml import TruncatedSVD as cuml_compressed_svd
from dask_ml.decomposition import IncrementalPCA as dask_pca
from cuml.decomposition import IncrementalPCA as cuml_pca
from cuml.metrics.trustworthiness import trustworthiness
from cuml import TSNE

from dask_ml.cluster import SpectralClustering


state_seed=23301522
state=np.random.RandomState(state_seed)

palette_discrete=lambda q:sns.color_palette("Set2", n_colors=q, desat=.9)
palette_contiguous=sns.color_palette("icefire", as_cmap=True)


def get_na_prop_df(x):
  if isinstance(x,pd.DataFrame):
    return x.apply(pd.isna).apply(pd.Series.value_counts,normalize=True)
  raise TypeError(f"{type(x)=}")


def interact_feature(darr,n=2)->tuple:
    if darr.ndim==2:
      poly=PolynomialFeatures(degree=n,interaction_only=True)
      Xp=poly.fit_transform(darr)
      
      return poly,Xp
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
  iterated_power=3,
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


def get_svd(darr,n=10,backend='cupy')->tuple:
  if backend=='dask':
    return da.linalg.svd_compressed(
      darr,
      k=n,
      compute=True
    )
  else:
    return cuml_compressed_svd(
      darr,
      algorithm = 'Jacobi',
      n_components = n,
      n_iter = 20
    )


def cluster_nx_iter(
    darr,
    cluster_count_range=range(2,11,1),
    component_count=100,
    affinity="polynomial",
    scoring=False,
    random_state=state
)->tuple:
  X=darr

  result={}
  for cluster_count in cluster_count_range:
    claim(
	    f"Attempting to cluster {X.shape}, {X.dtype}, {X.nbytes // 1024**2}MB for",
	    f"{cluster_count=}, {component_count=}"
    )
		
    t0=time()
    
    clusterer=SpectralClustering(
      n_clusters=cluster_count,
      n_components=component_count,
      affinity=affinity,
      degree=3,
      coef0=1,
      assign_labels="kmeans",
      persist_embedding=True,
      random_state=random_state,
      n_jobs=-1
    )

    clusterer.fit(X)
    claim(f"Clustring took {(time() - t0) // 1} s")

    if scoring:
      t0=time()
      score_historical=0
      Xl=clusterer.labels_
      score=trustworthiness(X,Xl,batch_size=384)

      if score>score_historical:
        score_historical=score
        clusterer_return=clusterer
      else:
        clusterer_return=None

      result_intermidiate=(score,clusterer_return)
      claim(f"Scoring took {(time() - t0) // 1} s")
    else:
      result_intermidiate=(1,clusterer)
    
    result[cluster_count]=result_intermidiate

  if scoring:
    fig,ax=plt.subplots(figsize=(6,6))

    ax.plot(cluster_count_range,[q[0] for q in result.values()])
    ax.set_xlabel("n")
    ax.set_ylabel("Score")
    ax.set_title(f"Score, {cluster_count_range=}")

    plt.savefig(
      f"./result/{str(time()).replace('.','')[:10]}-cluster_nx_iter.png",
      transparent=True
    )

    return result,fig,ax
  
  return result


def cluster_t_iter(
	X,
	label,
  param=dict(
    perplexity=[q for q in range(50, 1001, 50)],
    n_neighbors=[10 ** q for q in range(4,10)],
    n_iter=[10 ** q for q in range(2,5)],
    learning_rate=np.logspace(1.5,3,5)
  ),
  param_sample_size=3,
  random_state_seed=state_seed
):
  param_list=ParameterSampler(
    param,
    n_iter=param_sample_size,
  )

  result_total=[]
  for _param in param_list:
    claim(f"Fitting for ({_param=})")
    decomposer=TSNE(
      **_param,
      exaggeration_iter=int(np.sqrt(_param["n_iter"])),
      random_state=random_state_seed
    )

    result=decomposer.fit_transform(X)

    result_total.append((_param,result))

    if isinstance(result, cp.ndarray):
      result=result.get()

    cluster_plot(
      result,
      label,
      name="cluster_t_iter",
      figsize=(6,6)
    )

    result=None

  return result_total


def cluster_plot(
  Xr,
  label,
  name="cluster",
  figsize=(8,8),
  return_figure=True
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
        alpha=.8
      )

    ax.set_xlabel("Xr: X0")
    ax.set_ylabel("Xr: X1")

    plt.xticks([])
    plt.yticks([])
    ax.set_title("")

    plt.savefig(
      f"./result/{str(time()).replace('.','')[:13]}-{name}.png",
      transparent=True
    )

    if return_figure:
      return fig,ax
    else:
      plt.clf()
      return None

