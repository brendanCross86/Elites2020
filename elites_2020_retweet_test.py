import sqlite3
import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#from urllib.parse import urlparse
#from urllib.parse import unquote
#from PlotUtils import addDatetimeLabels, add_vspans
from GraphUtils import buildGraphCSV, loadRetweetGraphData, dataToGraphCSV
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

#from dask.diagnostics import ProgressBar
#from dask.distributed import Client, LocalCluster, progress

# number of workers
NUM_WORKERS = 64

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
#dask.config.set(scheduler='processes')

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
CLASSIFIED_URLS_DIR = '/home/pub/hernan/Election_2020/classified_links'
URLS_DIR = os.path.join(Election_2020_dir, 'urls')
WORKING_DIR = '/home/crossb/working'
media_types = ['left', 'right']#, 'central']

t00 = time.time()
t0 = time.time()
#
start_date = datetime(2020, 6, 1)
stop_date = datetime(2020, 11, 9)
edges_db_file = dict()


RETWEET_DIR_TO_COLUMNS = {
    os.path.join(Election_2020_dir, 'user_mentions'): ['id', 'user_id'],
    os.path.join(Election_2020_dir, 'retweets'): ['id', 'user_id', 'retweet_id'],
    os.path.join(Election_2020_dir, 'replies'): ['id', 'in_reply_to_user_id', 'in_reply_to_status_id'],
    os.path.join(Election_2020_dir, 'quotes'): ['id', 'user_id', 'quoted_id']
}


RENAME_COLS = {
    'id': 'id',
    'user_id': 'infl_id',
    'in_reply_to_user_id': 'infl_id',
    'retweet_id': 'infl_response_id',
    'in_reply_to_status_id': 'infl_response_id',
    'quoted_id': 'infl_response_id'
}


# get edges list

# set tmp dir
os.environ['SQLITE_TMPDIR'] = '/home/crossb'

# %% build graphs


def dask_load_tweets(data_dir):
    """

    :param data_dir:
    :param processes:
    :return:
    """

    # check the working directory for a copy of the data and load that instead
    if os.path.isfile(os.path.join(WORKING_DIR, 'date_filtered_tweets.csv')):
        print("LOADING WORKING TWEET DATA")
        #tweet_data = dd.read_csv(os.path.join(WORKING_DIR, 'date_filtered_tweets.csv'), delimiter=',',
        tweet_data = dd.read_csv(os.path.join(WORKING_DIR, 'tweets_less_duplicates_*.csv'), delimiter=',',
                    usecols=['id', 'timestamp', 'auth_id'], parse_dates=['timestamp'],
                    dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64})
                    #dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64})

    else:
        tweet_dir = os.path.join(data_dir, 'tweets')
        tweet_data = dd.read_csv(os.path.join(tweet_dir, '*.csv'), delimiter=',',
                                 dtype={'id': str, 'created_at': str, 'user_id': str}
                                 ).rename(columns={'user_id': 'auth_id', 'created_at': 'timestamp'})

        tweet_data = tweet_data[tweet_data['id'] != 'id']# %H:%M:%S')
        to_date = lambda df: df['timestamp'].str.slice(stop=10)
        tweet_data['timestamp'] = tweet_data.map_partitions(to_date)
        tweet_data['timestamp'] = dd.to_datetime(tweet_data['timestamp'], format='%Y-%m-%d')


        filter_dates = lambda df: df[(df['timestamp'] > start_date)&(df['timestamp'] < stop_date)]
        tweet_data = tweet_data.map_partitions(filter_dates)

    tweet_data = tweet_data.persist()
    progress(tweet_data)
    print("Load done")
    # get the number of partitions in the dataframe
    #n_partitions = tweet_data.npartitions
    #print("num partitions", n_partitions)
    #tweet_data = tweet_data.compute()
    #tweet_data = tweet_data.drop_duplicates()

    #tweet_data = dd.from_pandas(tweet_data, npartitions=n_partitions)
    tweet_data = tweet_data.set_index('id')
    tweet_data = tweet_data.persist()
    progress(tweet_data)
    print("Finished loading tweets!")
    #tweet_data.to_csv(os.path.join(WORKING_DIR, 'tweets_less_duplicates_*.csv'))
    return tweet_data


def correlate(directory, remove_duplicates=True):
    tweets = dask_load_tweets(Election_2020_dir)
    #directory = list(RETWEET_DIR_TO_COLUMNS.keys())[1]
    basename = os.path.basename(os.path.normpath(directory))
    print("Reading from {}".format(directory))
    stime = time.time()
    columns = RETWEET_DIR_TO_COLUMNS[directory]
    print("columns:", columns)
    data = dd.read_csv(os.path.join(directory, '*.csv'), delimiter=',', usecols=columns,
                       dtype={key: str for key in columns}).rename(
        columns={key: RENAME_COLS[key] for key in columns})
    print("Reading time taken: {}".format(time.time() - stime))
    # concat
    # pdb.set_trace()
    # data = data.merge(tweet_data, on=['id']).compute()
    convert_to_int = lambda df: pd.to_numeric(df, errors='coerce')
    for old_column in columns:
        column = RENAME_COLS[old_column]
        data[column] = data[column].map_partitions(convert_to_int)
    data = data.dropna()
    for old_column in columns:
        column = RENAME_COLS[old_column]
        data[column] = data[column].astype(np.int64)

    if remove_duplicates:
        print("Removing duplicates")
        pd_data = data.compute()
        pre_drop = len(pd_data)
        pd_data = pd_data.drop_duplicates()
        post_drop = len(pd_data)
        print("Rows before:", pre_drop)
        print("Rows after:", post_drop)
        print("Dropped {} duplicate entries!".format(pre_drop-post_drop))
        data = dd.from_pandas(pd_data, npartitions=data.npartitions)
        print("Dropped duplicates")


    # set our index to the id column, to make our merge with tweets faster
    stime = time.time()
    data = data.set_index('id')
    data = data.persist(retries=1000)
    progress(data)
    print("load time taken {} seconds".format(time.time() - stime))


    print("Merging Tweets and {}".format(basename))
    stime = time.time()
    data = data.merge(tweets, left_index=True, right_index=True).persist(retries=1000)
    progress(data)
    print("Merge time taken {} seconds".format(time.time() - stime))

    # remove any columns we don't need for our datetime edges
    print("Remove extra columns")
    stime = time.time()
    out_columns = ['id', 'infl_id', 'timestamp', 'auth_id']
    data = data.drop([x for x in data.columns if x not in out_columns], axis=1)
    # progress(data)
    data.to_csv(os.path.join(WORKING_DIR, 'merged_tweet_to_{}_*.csv'.format(basename)))
    print("Read and write of {} took {} seconds".format(directory, time.time() - stime))

    return


def dask_filter_stance():
    stance_dir = os.path.join(Election_2020_dir, 'classification')
    stime = time.time()
    stances_data = dd.read_csv(os.path.join(stance_dir, '*v2.txt'), delimiter=',',
                               header=None, usecols=[0, 1, 3], dtype={0: np.int64, 1: str, 3: str},
                               parse_dates=[1]).rename(columns={0: 'auth_id', 1: 'timestamp', 3: 'p'})

    stances_data['timestamp'] = dd.to_datetime(stances_data['timestamp'], format='%Y-%m-%d')
    filter_dates = lambda df: df[(df['timestamp'] > start_date) & (df['timestamp'] < stop_date)]
    stances_data = stances_data.map_partitions(filter_dates)

    convert_to_float = lambda df: pd.to_numeric(df, errors='coerce')
    stances_data['p'] = stances_data['p'].map_partitions(convert_to_float)
    stances_data = stances_data.dropna()
    #stances_data = stances_data[(stances_data['p'] > .66) & (stances_data < .33)]

    cols = list(stances_data.columns)
    cols.remove('p')
    print("Cols:", cols)
    stances_data = stances_data.groupby(cols).mean().reset_index().persist(retries=1000)

    #stances_data = stances_data.persist(retries=1000)
    progress(stances_data)
    print("Grouped stances columns = {}".format(list(stances_data.columns)))
    #stances_left = stances_data[stances_data['p'] < .33]
    stances_right = stances_data[stances_data['p'] > .66]
    #stances_left = stances_left.persist(retries=1000)
    stances_right = stances_right.persist(retries=1000)
    #progress(stances_left)
    progress(stances_right)
    print("Load and Filter Stance data took {} seconds".format(time.time() - stime))

    #stances_left.to_csv(os.path.join(WORKING_DIR, 'stance_data', 'left_stance_*.csv'), index=False)
    #stances_right.to_csv(os.path.join(WORKING_DIR, 'stance_data', 'right_stance_*.csv'), index=False)
    return stances_right#stances_left, stances_right


def dask_assign_stance(tweets, drop_duplicates=False, group_by=True):
    #s_left, s_right = dask_filter_stance()
    stances = ['left', 'right']
    for stance in stances:
        stance_dir = os.path.join(WORKING_DIR, 'stance_data', '{}_stance_0.csv'.format(stance))
        s_data = dd.read_csv(stance_dir, delimiter=',',
                             usecols=['auth_id', 'timestamp', 'p'],
                             dtype={'auth_id': np.int64, 'timestamp': str, 'p': np.float64},
                             parse_dates=[1])

        gb_cols = tweets.columns
        print("Begin merge tweets to stance {}".format(stance))
        stime = time.time()
        tweets = tweets.merge(s_data, on=['auth_id', 'timestamp']).persist(retries=1000)
        #if drop_duplicates:
        #    tweets = tweets.drop_duplicates()

        print("Combined edges in {} seconds".format(time.time() - stime))
        # split into two stances
        #del stances
        print("Splitting left / right")
        stime = time.time()

        edges_cols = ['id', 'auth_id', 'infl_id']
        #tweets = tweets[tweets['p'] < .33][edges_cols]
        #tweets = tweets[edges_cols]
        tweets = tweets.drop(['timestamp', 'p'], axis=1)
        tweets.persist(retries=1000)
        progress(tweets)

        tweets.to_csv(os.path.join(WORKING_DIR, '{}_edges.csv'.format(stance)), single_file=True, index=False)

        print("Split stances in {} seconds".format(time.time() - stime))
        #left.to_csv(os.path.join(WORKING_DIR, 'left_edges.csv'), single_file=True)
        #right.to_csv(os.path.join(WORKING_DIR, 'right_edges.csv'), single_file=True)

    return tweets


def dask_str_col_to_int(data, col):
    to_numeric = lambda df: pd.to_numeric(df, errors='coerce')
    data[col] = data[col].map_partitions(to_numeric)
    data = data.dropna()
    data[col] = data[col].astype(np.int64)
    return data


def assign_edge_classes(tweets):
    print("Begin assigning edge classes")
    stime_total = time.time()
    write_dir = '/home/crossb/working/url_classified_edgelists'

    # read the tweet classes
    urls_data = dd.read_csv(os.path.join(WORKING_DIR, 'tweet_classes.csv'), delimiter=',',
                            usecols=['id', 'leaning'],
                            dtype={'id': np.int64, 'leaning': str}
                            ).set_index('id')
    classes = urls_data['leaning'].unique()
    #print("Classes: {}".format(classes))
    for edge_class in classes:
        if edge_class in ['least', 'Right', 'Left', 'extreme right', 'Left-Center', 'extreme left', 'Right-Center']:
            continue
        print("Begin {} edgelist".format(edge_class))
        stime = time.time()
        class_data = urls_data[urls_data['leaning'] == edge_class]
        class_data.persist(retries=1000)
        progress(class_data)
        print("Split urls by leaning")
        # merge
        #class_data = tweets.merge(class_data, on=['id']).persist(retries=1000)
        class_data = tweets.merge(class_data, left_index=True, right_index=True).persist(retries=1000)
        progress(class_data)
        print("Merged Tweets to urls")
        class_data = class_data.drop(['timestamp', 'leaning'], axis=1)

        #progress(class_data)
        class_data.to_csv(os.path.join(write_dir, '{}_edges.csv'.format(edge_class)),
                          single_file=True)
        #class_data.to_csv(os.path.join(write_dir, '{}_edges.csv'.format(edge_class)), single_file=True, index=False)
        print("{} edgelist elapsed time {} seconds".format(edge_class, time.time() - stime))

    print("Assign edge classes elapsed time {} seconds".format(time.time() - stime_total))
    return


# Add methods for the creation of twitter
def tweet_to_class():
    print("Begin correlating tweets to stances!")
    stime = time.time()
    # READ THE URLS DATA
    urls_data = dd.read_csv(os.path.join(URLS_DIR, '*.csv'), delimiter=',',
                               usecols=['tweet_id', 'expanded_url'],
                               dtype={'tweet_id': str, 'expanded_url': str}
                            ).rename(columns={'tweet_id': 'id', 'expanded_url': 'url'})
    urls_data = dask_str_col_to_int(urls_data, 'id')

    # READ CLASSIFIED LINKS
    classifications = dd.read_csv(os.path.join(CLASSIFIED_URLS_DIR, '*.tsv'), delimiter='\t',
                                  usecols=['url', 'leaning'],
                                  dtype={'url': str, 'leaning': str}, quotechar='"')

    classifications = urls_data.merge(classifications, on='url')
    classifications = classifications.dropna().drop_duplicates()
    classifications = classifications.persist(retries=1000)
    progress(classifications)
    classifications = classifications.drop(['url'], axis=1)

    classifications.to_csv(os.path.join(WORKING_DIR, 'tweet_classes.csv'), index=False, single_file=True)
    print("Stance correlation took {} seconds".format(time.time() - stime))
    return


def debug_tweet_classes():
    """
    Confirm that nothing weird is happening with our resulting networks
    This check was inspired by the fact that Trump is the most influential node (by pagerank)
    in both the Left and Right networks. Confirm edges are going where they belong

    :return:
    """
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2,
                           scheduler_port=0, dashboard_address=None)
    client = Client(cluster)
    print("Begin debugging")
    urls_data = dd.read_csv(os.path.join(URLS_DIR, '*.csv'), delimiter=',',
                            usecols=['tweet_id', 'user_id', 'expanded_url'],
                            dtype={'tweet_id': str, 'user_id': str, 'expanded_url': str}
                            ).rename(columns={'tweet_id': 'id', 'expanded_url': 'url'})
    urls_data = dask_str_col_to_int(urls_data, 'id')
    urls_data = dask_str_col_to_int(urls_data, 'user_id')

    classifications = dd.read_csv(os.path.join(CLASSIFIED_URLS_DIR, '*.tsv'), delimiter='\t',
                                  usecols=['url', 'leaning'],
                                  dtype={'url': str, 'leaning': str}, quotechar='"')

    merged = urls_data.merge(classifications, on='url').persist(retries=1000)
    progress(merged)
    pdb.set_trace()

    return


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

    # Step 1: Process/Load our twitter data and correlate that data to our twitter responses
    #for dir in [os.path.join(Election_2020_dir, 'retweets'),
    #            os.path.join(Election_2020_dir, 'replies'),
    #            os.path.join(Election_2020_dir, 'quotes')]:
    #    correlate(dir)
    ## user mentions is formatted without a corresponding response retweet/reply, etc. id, so we will process
    ## differently
    #correlate(os.path.join(Election_2020_dir, 'user_mentions'), remove_duplicates=False)

    # Step 2: Concatenate all the edges from step 1 into one edgelist
    edges = concat_edges()

    # Optional step 3: Classify edges by stance
    # write to file
    #dask_filter_stance()

    # step 4: Classify edges by political leaning
    ## URLS ##
    #tweet_to_class()
    #assign_edge_classes(edges)

    ## STANCES ##
    # separate by stances
    dask_assign_stance(edges)



    # load our
    edges_array = []

    # save
    with open(os.path.join(save_dir, 'edge_lists.pickle'), 'wb') as fopen:
        #pickle.dump(edges_db_file, fopen)
        pickle.dump(edges_array, fopen)

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


def concat_edges(write=False):
    stime = time.time()
    columns = ['id', 'infl_id', 'timestamp', 'auth_id']
    data = dd.read_csv(os.path.join(WORKING_DIR, 'merged_tweet*.csv'), delimiter=',', usecols=columns,
                       dtype={'id': np.int64, 'infl_id': np.int64, 'timestamp': str, 'auth_id': np.int64},
                       parse_dates=['timestamp'],).set_index('id')

    data['timestamp'] = dd.to_datetime(data['timestamp'], format='%Y-%m-%d')
    # write one large edgelist file before splitting
    data.persist(retries=1000)
    progress(data)
    if write:
        data.to_csv(os.path.join(WORKING_DIR, 'all_edges.csv'), single_file=True)
    print("Combined edges in {} seconds".format(time.time() - stime))
    # stance stuff

    return data


def main():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2,
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

