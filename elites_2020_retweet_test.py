import sqlite3
import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#from urllib.parse import urlparse
#from urllib.parse import unquote
#from PlotUtils import addDatetimeLabels, add_vspans
from GraphUtils import buildGraphCSV, loadRetweetGraphData, dataToGraphCSV, political_stance_query
import GraphUtils as gu
from datetime import datetime
from subprocess import check_output

import time
import pickle
import graph_tool.all as gt
from multiprocessing import Pool
from functools import partial
import pdb
import collections
import dask.dataframe as dd
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, progress

# number of workers
NUM_WORKERS = 32

#cpu_mask = os.sched_getaffinity(os.getpid())
#print("initial affinity mask: {}".format(cpu_mask))
#working_cores = [x for x in range(64)]
#os.sched_setaffinity(os.getpid(), working_cores)
#updated_cpu_mask = os.sched_getaffinity(os.getpid())
#print("updated affinity mask: {}".format(updated_cpu_mask))

#pdb.set_trace()


# set up local cluster for dask
#cluster = LocalCluster(n_workers=NUM_WORKERS, scheduler_port=0, dashboard_address=None, worker_dashboard_address=None)
#client = Client(cluster)
dask.config.set(scheduler='processes')

#from TwStats import chunks

#from PlotUtils import compute_CCDF

#save_dir = '../data/urls/revisions/'
save_dir = 'data/test2020'

#raise Exception

#%% load user and tweet list
#tweet_db_file1 = '../databases_ssd/complete_trump_vs_hillary_db.sqlite'
#tweet_db_file2 = '../databases_ssd/complete_trump_vs_hillary_sep-nov_db.sqlite'
#urls_db_file = '../databases_ssd/urls_db.sqlite'
tweet_db_file1 = '/home/pub/hernan/Election_2016/complete_trump_vs_hillary_db.sqlite'
tweet_db_file2 = '/home/pub/hernan/Election_2016/complete_trump_vs_hillary_sep-nov_db.sqlite'
# The following code is for 2020 CSV data
Election_2020_dir = '/home/pub/hernan/Election_2020/joined_output'
WORKING_DIR = '/home/crossb/working'
media_types = ['left', 'right']#, 'central']
media_type_to_query = {
    'left': "SELECT tweet_id FROM class_proba WHERE p_pro_hillary_anti_trump > .5",
    'right': "SELECT tweet_id FROM class_proba WHERE p_pro_hillary_anti_trump < .5"
}

t00 = time.time()
t0 = time.time()
#
start_date = datetime(2020, 6, 1)
stop_date = datetime(2020, 11, 9)
edges_db_file = dict()


RETWEET_DIR_TO_COLUMNS = {
    os.path.join(Election_2020_dir, 'user_mentions'): ['id', 'user_id'],
    os.path.join(Election_2020_dir, 'retweets'): ['id', 'retweet_id'],
    os.path.join(Election_2020_dir, 'replies'): ['id', 'in_reply_to_status_id'],
    os.path.join(Election_2020_dir, 'quotes'): ['id', 'quoted_id']
}

RENAME_COLS = {
    'id': 'id',
    'user_id': 'infl_id',
    'retweet_id': 'infl_id',
    'in_reply_to_status_id': 'infl_id',
    'quoted_id': 'infl_id'
}


# get edges list

# set tmp dir
os.environ['SQLITE_TMPDIR'] = '/home/crossb'


def dask_load_tweets(data_dir, processes):
    """

    :param data_dir:
    :param processes:
    :return:
    """

    # check the working directory for a copy of the data and load that instead
    if os.path.isfile(os.path.join(WORKING_DIR, 'date_filtered_tweets.csv')):
        print("LOADING WORKING TWEET DATA")
        tweet_data = dd.read_csv(os.path.join(WORKING_DIR, 'date_filtered_tweets.csv'), delimiter=',',
                    usecols=['id', 'timestamp', 'auth_id'], parse_dates=['timestamp'],
                    dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64})
                    #dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64})

    else:
        tweet_dir = os.path.join(data_dir, 'tweets')
        tweet_data = dd.read_csv(os.path.join(tweet_dir, '*.csv'), delimiter=',',
                                 dtype={'id': str, 'created_at': str, 'user_id': str}
                                 ).rename(columns={'user_id': 'auth_id','created_at': 'timestamp'})

        tweet_data = tweet_data[tweet_data['id'] != 'id']# %H:%M:%S')
        to_date = lambda df: df['timestamp'].str.slice(stop=10)
        tweet_data['timestamp'] = tweet_data.map_partitions(to_date)
        tweet_data['timestamp'] = dd.to_datetime(tweet_data['timestamp'], format='%Y-%m-%d')


        filter_dates = lambda df: df[(df['timestamp'] > start_date)&(df['timestamp'] < stop_date)]
        tweet_data = tweet_data.map_partitions(filter_dates)

    tweet_data = tweet_data.persist()
    #progress(tweet_data)

    return tweet_data


def dask_filter_stance(stance_dir):
    stime = time.time()
    stances_data = dd.read_csv(os.path.join(stance_dir, '*v2.txt'), delimiter=',',
                               header=None, usecols=[0, 1, 3], dtype={0: np.int64, 1: str, 3: str},
                               parse_dates=[1]).rename(columns={0: 'auth_id', 1: 'timestamp', 3: 'p'})

    stances_data['timestamp'] = dd.to_datetime(stances_data['timestamp'], format='%Y-%m-%d')
    filter_dates = lambda df: df[(df['timestamp'] > start_date) & (df['timestamp'] < stop_date)]
    stances_data = stances_data.map_partitions(filter_dates)
    #convert_to_float = lambda df: df[df['p'].str.isnumeric()]
    convert_to_float = lambda df: pd.to_numeric(df, errors='coerce')
    stances_data['p'] = stances_data['p'].map_partitions(convert_to_float)
    stances_data = stances_data.dropna()
    #stances_data = stances_data.map_partitions(convert_to_float)
    #stances_data['p'] = stances_data['p'].astype(float)

    stances_data = stances_data.persist()
    #progress(stances_data)

    print("Load and Filter Stance data took {} seconds".format(time.time() - stime))
    return stances_data


def dask_assign_stance(tweets, stances):
    tweets = tweets.merge(stances, on=['auth_id', 'timestamp'], how='left').set_index('id').persist(retries=1000)
    #.persist(retries=1000)#.set_index('id').persist()
    #progress(tweets)

    return tweets


def dask_gather_edges(tweet_data):
    influence_data = []
    for directory in RETWEET_DIR_TO_COLUMNS.keys():
        basename = os.path.basename(os.path.normpath(directory))
        print("Basename:", basename)
        print("Reading from {}".format(directory))
        stime = time.time()
        columns = RETWEET_DIR_TO_COLUMNS[directory]
        data = dd.read_csv(os.path.join(directory, '*.csv'), delimiter=',', usecols=columns,
                           dtype={key: str for key in columns},).rename(
            columns={key: RENAME_COLS[key] for key in RETWEET_DIR_TO_COLUMNS[directory]})
        print("Reading time taken: {}".format(time.time() - stime))
        # concat
        #pdb.set_trace()
        #data = data.merge(tweet_data, on=['id']).compute()
        convert_to_int = lambda df: pd.to_numeric(df, errors='coerce')
        for old_column in columns:
            column = RENAME_COLS[old_column]
            data[column] = data[column].map_partitions(convert_to_int)
        data = data.dropna()
        for old_column in columns:
            column = RENAME_COLS[old_column]
            data[column] = data[column].astype(np.int64)
        data = data.set_index('id')

        stime = time.time()
        data = data.persist(retries=1000)
        #progress(data)
        print("Load {} took {} seconds".format(basename, time.time() - stime))

        stime = time.time()
        #data = data.merge(tweet_data, on=['id']).persist(retries=1000)
        data = data.merge(tweet_data, left_index=True, right_index=True).persist(retries=1000)
        #progress(data)
        data.to_csv(os.path.join(WORKING_DIR, 'merged_tweet_to_{}_*.csv'.format(basename)))
        print("Read and write of {} took {} seconds".format(directory, time.time() - stime))
        # to save on time and memory we will write each set of edges to their own files and then add
        # them to the appropriate networks separately


        #influence_data.append(data)#data.merge(tweet_data, on=['id']))
        #influence_data.append(data)
        #print("Appending thing")

    #influence_data = dd.concat(influence_data)
    #tweet_data = tweet_data.merge(influence_data, on=['id'])

    #influence_data = influence_data.persist(retries=1000)
    #progress(influence_data)

    return influence_data


# %% build graphs

def build_graph_from_edges(edge_list, graph_name):
    G = gt.Graph(directed=True)
    G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
    G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')
    G.edge_properties['source_id'] = G.new_edge_property('int64_t')
    G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id, G.ep.source_id])

    G.gp['name'] = G.new_graph_property('string', graph_name)

    return G


def add_vertex_properties(G):
    # compute some vertex properties

    G.vp['k_out'] = G.degree_property_map('out')
    G.vp['k_in'] = G.degree_property_map('in')


def gather_edges():
    print("Begin gathering edge data")


    # first step is to check if intermediate results exist and load the most recent

    start_ttime = time.time()
    stime = time.time()
    filtered_tweets = dask_load_tweets(Election_2020_dir, NUM_WORKERS)
    print("Load tweets time taken: {}".format(time.time() - stime))
    #stime = time.time()
    #filtered_tweets.to_csv(os.path.join(WORKING_DIR, 'date_filtered_tweets.csv'), single_file=True)
    #progress(filtered_tweets)
    #print("Write tweets took {} seconds".format(time.time() - stime))

    # filtered_tweets = preprocess_data(Election_2020_dir, 12)
    # unique_tweet_ids = set([row[0] for row in filtered_tweets])
    # write to file

    stance_dir = os.path.join(Election_2020_dir, 'classification')

    # get tweets by political stance
    # left_tweets, right_tweets = political_stance_query(Election_2020_dir, threads=5)
    # load our csv data in to pandas dataframe

    stime = time.time()
    filtered_tweets = dask_assign_stance(filtered_tweets, dask_filter_stance(stance_dir))
    print("Stance Assignment took {} seconds".format(time.time() - stime))

    stime = time.time()
    filtered_tweets = dask_gather_edges(filtered_tweets)[['id', 'auth_id', 'infl_id', 'p']]
    print("Gather Edges time taken: {} seconds".format(time.time() - stime))

    pdb.set_trace()
    # write the gathered edges to file
    filtered_tweets.to_csv(os.path.join(WORKING_DIR, 'retweet_edges_all_*.csv'))
    progress(filtered_tweets)
    # split by stance and build both graphs
    stime = time.time()


    # run our delayed operations.
    #dask.compute()

    edge_cols = list(filtered_tweets.columns[:3])
    #separate_stances = lambda df: (df[df['p'] < .33][edge_cols], df[df['p'] > .66][edge_cols])
    #left_data, right_data = dask.persist(separate_stances(filtered_tweets))
    left_data, right_data = dask.persist(filtered_tweets[filtered_tweets['p'] < .33][edge_cols],
                                         filtered_tweets[filtered_tweets['p'] > .66][edge_cols])
    progress(left_data, right_data)
    pdb.set_trace()
    #left_data = edges[edges['p'] < .33][edge_cols]
    #right_data = edges[edges['p'] > .66][edge_cols]

    #left_data.compute()
    #right_data.compute()
    print("computed time: {} seconds".format(time.time() - stime))
    stime = time.time()
    edges_array = {'left': left_data.to_records(), 'right': right_data.to_records()}
    #edges_array = {
    #    'left': dataToGraphCSV(left_data.values(), 'retweet', graph_lib='edgelist'),
    #    'right': dataToGraphCSV(right_data.values(), 'retweet', graph_lib='edgelist')
    #}
    print("Build Graphs time taken: {}".format(time.time() - stime))

    # pdb.set_trace()
    # convert our full dataset into two datasets by political stance
    # left_data = data[data['id'].isin(left_tweets['id'])]
    # right_data = data[data['id'].isin(right_tweets['id'])]

    # edges_db_file['left'] = dataToGraphCSV(left_data, 'retweet', graph_lib='edge_list')
    # edges_db_file['right'] = dataToGraphCSV(right_data, 'retweet', graph_lib='edge_list')


    #edges_array = buildGraphCSV(Election_2020_dir, 'retweet', start_date, stop_date, graph_lib='edge_list')


    # save
    with open(os.path.join(save_dir, 'edge_lists.pickle'), 'wb') as fopen:
        #pickle.dump(edges_db_file, fopen)
        pickle.dump(edges_array, fopen)

    print("Gather Edges time taken: {} seconds".format(time.time() - start_ttime))
    return edges_array


def build_graph(edges_array):
    # %%
    # build and process each graph
    t0 = time.time()
    for stance in ['left', 'right']:
        for graph_type in ['complete', 'simple']:
            for period in ['june-nov']:
                print(stance)
                print(graph_type)
                print(period)

                edges_array = edges_db_file[stance]

                print(len(edges_array))
                graph_name = '2020_test_{}_{}'.format(stance, graph_type)
                G = build_graph_from_edges(edges_array, graph_name)

                if graph_type == 'simple':
                    gt.remove_parallel_edges(G)
                    gt.remove_self_loops(G)

                G.gp['period'] = G.new_graph_property('string', period)
                G.gp['graph_type'] = G.new_graph_property('string', graph_type)

                print(time.time() - t0)

                add_vertex_properties(G)
                print(time.time() - t0)

                # save graph
                print('saving graph to')
                filename = os.path.join(save_dir, 'retweet_graph_' + stance + '_' +
                                        graph_type + '_' + period + '.gt')
                print(filename)
                G.save(filename)
                print(time.time() - t0)

    print('total time')
    print(time.time() - t00)
    return


def main():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=4,
                           scheduler_port=0, dashboard_address=None)
    client = Client(cluster)
    print("Starting Retweet Network Generation")
    outer_stime = time.time()
    edges_array = gather_edges()
    print("Edges gathered in {} seconds".format(time.time() - outer_stime))
    stime = time.time()
    #build_graph(edges_array)
    print("Retweet networks built in {} seconds".format(time.time() - stime))
    print("Generate Retweet Networks Elapsed Time: {} seconds".format(time.time() - outer_stime))
    return


if __name__ == '__main__':
    main()

