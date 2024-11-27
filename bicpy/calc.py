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
from dask_ml.decomposition import IncrementalPCA
from cuml.metrics.trustworthiness import trustworthiness
from cuml import TSNE

from dask_ml.cluster import SpectralClustering

seed=23301522
state=np.random.RandomState(seed)

palette_discrete=sns.color_palette("Set1", desat=.8)
palette_discrete_many=palette_discrete
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

def pca(darr,n=2,batch_size=300)->tuple:
    decomposer=IncrementalPCA(n_components=n,batch_size=batch_size,random_state=state)
    return decomposer,decomposer.fit_transform(darr)

def get_svd(darr,n=10)->tuple:
    return tuple(q.compute() for q in da.linalg.svd_compressed(darr,k=n))

def cluster_nx(
    darr,
    cluster_count_range=range(2,101,2),
    component_count=100,
    affinity="polynomial",
    degree=3,
    random_state=state
)->tuple:
  X=darr

  result={}
  score_historical=0
  for cluster_count in cluster_count_range:
    component_count=100

    claim(f"Attempting to cluster {X.shape}, {X.dtype}, {X.nbytes // 1024**2}MB for",f"{cluster_count=}, {component_count=}")
    t0=time()
    
    clusterer=SpectralClustering(
      n_clusters=cluster_count,
      n_components=component_count,
      affinity=affinity,
      degree=degree,
      assign_labels="kmeans",
      persist_embedding=True,
      random_state=random_state
    )

    clusterer.fit(X)
    claim(f"Clustring took {(time() - t0) // 1} s")

    t0=time()
    Xl=clusterer.labels_
    score=trustworthiness(X,Xl,batch_size=384)

    if score>score_historical:
      score_historical=score
      clusterer_return=clusterer
    else:
      clusterer_return=None

    result[cluster_count]=(score,clusterer_return)
    claim(f"Scoring took {(time() - t0) // 1} s")

  fig,ax=plt.subplots(figsize=(8,8))

  ax.plot(cluster_count_range,[q[0] for q in result.values()])
  ax.set_xlabel("n")
  ax.set_ylabel("Score")
  ax.set_title(f"Score, {cluster_count_range=}")
  plt.savefig(f"./cluster_nx_trustworthiness-{str(time().replace(".",""))[:10]}.png",transparent=True)

  return result,fig,ax

def clustering_t(
	darr,
	label_groundtruth,
	param_distributions=dict(
		perplexity=range(5,201,10),
		learning_rate=range(1,201,10),
		n_neighbors=range(30,601,15),
		n_iter=[10**q for q in range(1,5)],
		metric=["l1", "cityblock", "manhattan", "euclidean", "l2", "cosine", "correlation"],
	),
	n=10**3,
	random_state=state
):
	X=darr
    
	param_distribution=ParameterSampler(param_distributions,n,random_state=random_state)
  
	result_total=[]
	for param in param_distribution:
		print(f"TSNE: Fitting for ({param=})")
		decomposer=TSNE(random_state=random_state,**param)
		result=decomposer.fit_transform(X)
		result_total.append((param,result))
    
	return result_total

def cluster_plot(darr,obs_label):
  Xr=darr.compute()

  if Xr.ndim==2:
    Xl=da.unique(obs_label).compute()
    fig,ax=plt.subplots(figsize=(8,8))

    for color,label in zip(palette_discrete_many(Xl.size),Xl):
      plt.scatter(
        Xr[obs_label==label, 0],
        Xr[obs_label==label, 1],
        color=color,
        label=str(label),
        alpha=.7
      )

    ax.set_xlabel("Xr: x0")
    ax.set_ylabel("Xr: x1")
    ax.set_title(f"Clusters by ({Xl.size}) length of Xl")
    plt.savefig(f"./clusters-{str(time().replace(".",""))[:10]}.png",transparent=True)

    return fig,ax
