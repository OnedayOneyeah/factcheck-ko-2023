import tqdm
import time, os, json
import numpy as np
import pandas as pd
from multiprocessing import Pool # for parallelizatin
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


with open(os.path.join(os.path.abspath(os.path.dirname('__main__')), "data/wiki_claims.json"), "r") as fp:
        dataset = json.load(fp)

# multiprocessing: parallelizing
df = pd.DataFrame(dataset)
df_filtered = df.iloc[:,6:13].drop('Date', axis = 1)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
corpus = list(set(df_filtered.claim))

def work_func(data):
    print('PID :', os.getpid()) # get the process id
    

    return data

def parallel_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    print(len(df_split))
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def main():
    start = int(time.time())
    num_cores = 8
    df = df_chloe
    #print(f'#1 : {df}')
    df = parallel_dataframe(df, work_func, num_cores)
    print(f'#2 : {df}')
    print('***run time(sec) : ', int(time.time()) - start)

if __name__ == '__main__':
    main()