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
from dask.distributed import Client, LocalCluster, progress
from multiprocessing.pool import ThreadPool
from dask.diagnostics import ProgressBar
#dask.config.set(scheduler='processes')
# number of workers
NUM_WORKERS = 32
#dask.config.set(scheduler='processes', pool=ThreadPool(NUM_WORKERS))

save_dir = 'data/test2020'


Election_2020_dir = '/home/pub/hernan/Election_2020/joined_output_v2'
CLASSIFIED_URLS_DIR = '/home/pub/hernan/Election_2020/classified_links'
URLS_DIR = os.path.join(Election_2020_dir, 'urls')
WORKING_DIR = '/home/crossb/working'
media_types = ['left', 'right']#, 'central']

t00 = time.time()
t0 = time.time()
#
start_date = datetime(2020, 6, 1)
stop_date = datetime(2020, 11, 3)
edges_db_file = dict()


RETWEET_DIR_TO_COLUMNS = {
    os.path.join(Election_2020_dir, 'user_mentions'): ['id', 'user_id'],
    os.path.join(Election_2020_dir, 'retweets'): ['id', 'user_id'],#, 'retweet_id'],
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

# set tmp dir
os.environ['SQLITE_TMPDIR'] = '/home/crossb'

# %% build graphs
# TODO: Break  load tweets into two portions, one where we load from raw and filter out by date, then save
#  and a second where we load the date filtered data and remove duplicates (since removing duplicates seems to
#  cause a host of issues as the amount of data increases)
def dask_load_tweets(tweet_dir):
    """

    :param data_dir:
    :param processes:
    :return:
    """

    # check the working directory for a copy of the data and load that instead
    #if os.path.isfile(os.path.join(WORKING_DIR, 'intermediaries', 'date_filtered_tweets.csv')):
    #try:
    #    print("LOADING WORKING TWEET DATA")
    #    files = os.path.join(WORKING_DIR, 'intermediaries', 'tweets_less_duplicates_*.csv')
    #    tweet_data = dd.read_csv(files, delimiter=',',
    #                usecols=['id', 'timestamp', 'auth_id'], parse_dates=['timestamp'],
    #                dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64})
#
    #except OSError as e:
    print("No working set of tweets found, filtering from raw data. Directory:", tweet_dir)
    tweet_data = dd.read_csv(os.path.join(tweet_dir, '*.csv'), delimiter=',',
                             dtype={'id': np.int64, 'created_at': str, 'user_id': np.int64, 'p': np.float64}
                             ).rename(columns={'user_id': 'auth_id', 'created_at': 'timestamp'})

    tweet_data = tweet_data[tweet_data['id'] != 'id']
    tweet_data['timestamp'] = dd.to_datetime(tweet_data['timestamp'], format='%Y-%m-%d %H:%M:%S')

    filter_dates = lambda df: df[(df['timestamp'] > start_date) & (df['timestamp'] < stop_date)]
    tweet_data = tweet_data.map_partitions(filter_dates)
    tweet_data = tweet_data.persist()
    progress(tweet_data)
    print("Filtered tweets by date range.")

    #print()
    #print("Saving filtered tweet data.")
    #tweet_data.to_csv(os.path.join(WORKING_DIR, 'intermediaries', 'date_filtered_tweets_june_to_election_*.csv'))
    #tweet_data = dask_str_col_to_int(tweet_data, ['id', 'auth_id'])

    # drop dupes
    #tweet_data = tweet_data.drop_duplicates().persist(retries=100)


    #tweet_data = tweet_data.persist(retries=100)
    #progress(tweet_data); print("loaded tweets!")

    # get the number of partitions in the dataframe
    n_partitions = tweet_data.npartitions
    print("num partitions", n_partitions)
    tweet_data = tweet_data.compute()
    print("We have {} tweets before removing duplicates".format(len(tweet_data)))
    tweet_data = tweet_data.drop_duplicates(subset=['id'])
    print("We have {} tweets after removing duplicates".format(len(tweet_data)))
    tweet_data = dd.from_pandas(tweet_data, npartitions=n_partitions).persist(retries=1000)
    #tweet_data = tweet_data.drop_duplicates().persist(retries=1000)
    progress(tweet_data); print("Dropped duplicates")
    tweet_data = tweet_data.set_index('id').persist(retries=100)
    progress(tweet_data)
    #tweet_data.to_csv(os.path.join(WORKING_DIR, 'intermediaries', 'tweets_less_duplicates_new_file_*.csv'))

    print("Finished loading tweets!")
    return tweet_data


def gather_retweet_edges_v2(tweet_dir, retweet_dir):
    """
    This method does the job of load tweets and correlate but for our newly corrected data, whose format
    has been slightly altered
    :return:
    """

    # load tweets
    #tweet_data = dd.read_csv(os.path.join(tweet_dir, '*.csv'), delimiter=',', usecols=['tweet_id', 'created_at'],
    #                         dtype={'tweet_id': str, 'created_at': str}
    #                         ).rename(columns={'tweet_id': 'id',  'created_at': 'timestamp'})
    tweet_data = dd.read_csv(os.path.join(tweet_dir, '*.csv'), delimiter=',', usecols=['id', 'p', 'timestamp'],
                             dtype={'id': np.int64, 'p': np.float64, 'timestamp': str})

    #tweet_data = tweet_data[tweet_data['id'] != 'id']
    tweet_data['timestamp'] = dd.to_datetime(tweet_data['timestamp'], format='%Y-%m-%d %H:%M:%S')

    filter_dates = lambda df: df[(df['timestamp'] > start_date) & (df['timestamp'] < stop_date)]
    tweet_data = tweet_data.map_partitions(filter_dates)
    tweet_data = tweet_data.persist()
    progress(tweet_data); print("Tweets: Filtered tweets on datetime")
    # convert id to int and set as index
    #tweet_data = dask_str_col_to_int(tweet_data, 'id').persist(retries=100)
    #progress(tweet_data); print("Tweets: Convert id to int")

    # remove duplicates
    partitions = tweet_data.npartitions
    tweet_data = tweet_data.drop_duplicates(subset='id').repartition(npartitions=partitions).persist(retries=100)
    progress(tweet_data); print("Tweets: Dropped duplicates")

    # set index
    tweet_data = tweet_data.set_index('id').persist(retries=100)
    progress(tweet_data); print("Tweets: Set index to id")


    # load retweets
    retweet_data = dd.read_csv(os.path.join(retweet_dir, '*.csv'), delimiter=',', usecols=['tweet_id', 'auth_id', 'infl_id'],
                             dtype={'tweet_id': str, 'auth_id': str, 'infl_id': str}
                             ).rename(columns={'tweet_id': 'id', 'user_id': 'auth_id'})
    # convert string columns to int
    retweet_data = dask_str_col_to_int(retweet_data, 'id').persist(retries=100)
    progress(retweet_data); print("Retweets: Converted id to int")

    # remove duplicates
    partitions = retweet_data.npartitions
    retweet_data = retweet_data.drop_duplicates(subset='id').repartition(npartitions=partitions).persist(retries=100)
    progress(retweet_data); print("Retweets: Dropped duplicates")

    # set index
    retweet_data = retweet_data.set_index('id').persist(retries=100)
    progress(retweet_data); print("Retweets: Set index to id")



    # join the data
    print("Merging tweets to retweets")
    intermediaries = retweet_data.merge(tweet_data, left_index=True, right_index=True).persist(retries=100)
    progress(intermediaries); print("Merged tweets and retweets")


    print("Writing Filtered Retweet intermediaries")
    write_dir = os.path.join(WORKING_DIR, 'stance_merged_retweets')
    intermediaries.to_csv(os.path.join(write_dir, 'merged_retweets_alldata_*.csv'))

    return


def drop_duplicate_rows(files):
    """
    Given a list of files, remov
    :param files:
    :return:
    """
    for dir_path, file in files:
        out_filename = "less_duplicates_" + file
        data = dd.read_csv(file, delimiter=',',
                           usecols=['id', 'timestamp', 'auth_id'], parse_dates=['timestamp'],
                           dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64})

    return


def correlate(directory, remove_duplicates=True):
    #tweets = dask_load_tweets(os.path.join(Election_2020_dir, 'tweets'))
    tweets = dask_load_tweets(os.path.join(WORKING_DIR, 'stance_merged_tweets'))
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
        #print("Length of retweets before: {}".format(len(data)))
        #data = data.drop_duplicates().persist(retries=100)
        #progress(data); print("Finished dropping dupes")
        #print("Length of retweets after: {}".format(len(data)))
        pd_data = data.compute()
        pre_drop = len(pd_data)
        print("Rows before:", pre_drop)
        pd_data = pd_data.drop_duplicates(subset=['id'])
        post_drop = len(pd_data)
        print("Rows after:", post_drop)
        print("Dropped {} duplicate entries!".format(pre_drop-post_drop))
        print("Are there duplicates left in {} data? {}".format(basename, pd_data.duplicated().any()))
        data = dd.from_pandas(pd_data, npartitions=data.npartitions).persist(retries=100)
        progress(data)
        print("Dropped duplicates")

    # set our index to the id column, to make our merge with tweets faster
    stime = time.time()
    data = data.set_index('id').persist(retries=1000)
    #data = data
    progress(data)
    print("load time taken {} seconds".format(time.time() - stime))

    # To merge, we will concat merge for speed
    # first get all intersecting ids
    print("Merging Tweets and {}".format(basename))
    stime = time.time()
    #tweets_in_both = data.index.intersection(tweets.index)
    tweets_in_both = np.intersect1d(np.array(data.index), np.array(tweets.index))
    data = data.loc[tweets_in_both]
    tweets = tweets.loc[tweets_in_both]
    tweets['infl_id'] = data['infl_id']

    #data = data.merge(tweets, left_index=True, right_index=True).persist(retries=1000)
    #progress(data)
    print("Merge time taken {} seconds".format(time.time() - stime))

    # remove any columns we don't need for our datetime edges
    print("Remove extra columns")
    stime = time.time()
    out_columns = ['id', 'infl_id', 'timestamp', 'auth_id', 'p']
    tweets = tweets.drop([x for x in data.columns if x not in out_columns], axis=1).persist(retries=100)
    tweets = progress(tweets)

    #data.to_csv(os.path.join(WORKING_DIR, 'intermediaries', 'merged_tweet_to_{}_*.csv'.format(basename)))
    tweets.to_csv(os.path.join(WORKING_DIR, 'test', 'merged_tweet_to_{}_*.csv'.format(basename)))
    print("Read and write of {} took {} seconds".format(directory, time.time() - stime))

    return


def dask_filter_stance():
    stance_dir = os.path.join('/home/pub/hernan/Election_2020/joined_output/classification')
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
    #stances_right = stances_data[stances_data['p'] > .66]
    #stances_left = stances_left.persist(retries=1000)
    #stances_right = stances_right.persist(retries=1000)
    #progress(stances_left)
    #progress(stances_right)
    print("Load and Filter Stance data took {} seconds".format(time.time() - stime))

    #stances_left.to_csv(os.path.join(WORKING_DIR, 'stance_data', 'left_stance_*.csv'), index=False)
    #stances_right.to_csv(os.path.join(WORKING_DIR, 'stance_data', 'right_stance_*.csv'), index=False)
    return stances_data#stances_left, stances_right


def dask_assign_stance(tweets):
    #s_left, s_right = dask_filter_stance()
    stances = ['left', 'right']
    for stance in stances:
        stance_dir = os.path.join(WORKING_DIR, 'stance_data', '{}_stance_0.csv'.format(stance))
        s_data = dd.read_csv(stance_dir, delimiter=',',
                             usecols=['auth_id', 'timestamp', 'p'],
                             dtype={'auth_id': np.int64, 'timestamp': str, 'p': np.float64},
                             parse_dates=[1])
        s_data['timestamp'] = dd.to_datetime(s_data['timestamp'], format='%Y-%m-%d').persist(retries=1000)
        progress(s_data)

        print("Begin merge tweets to stance {}".format(stance))
        stime = time.time()
        s_data = tweets.merge(s_data, on=['auth_id', 'timestamp']).persist(retries=1000)
        progress(tweets)

        print("Combined edges in {} seconds".format(time.time() - stime))
        # split into two stances
        #del stances
        print("Splitting left / right")
        stime = time.time()

        edges_cols = ['id', 'auth_id', 'infl_id']
        s_data = s_data.drop(['timestamp', 'p'], axis=1)
        s_data.persist(retries=1000)
        progress(s_data)

        s_data.to_csv(os.path.join(WORKING_DIR, '{}_edges.csv'.format(stance)), single_file=True)#, index=False)

        print("Split stances in {} seconds".format(time.time() - stime))
        #left.to_csv(os.path.join(WORKING_DIR, 'left_edges.csv'), single_file=True)
        #right.to_csv(os.path.join(WORKING_DIR, 'right_edges.csv'), single_file=True)

    return tweets


def write_stance_edgelists(intermediaries_path):
    print("Writing stance edgelist, using intermediaries from", intermediaries_path)
    stances_data = dd.read_csv(os.path.join(intermediaries_path, '*.csv'), delimiter=',',
                               dtype={'id': np.int64, 'timestamp': str, 'user_id': np.int64,
                                      'p': np.float64, 'infl_id': np.int64}).set_index('id').persist(retries=100)
    progress(stances_data); print("Finished loading stance intermediaries")
    stances_left = stances_data[stances_data['p'] < .33].drop(['timestamp', 'p'], axis=1).persist(retries=100)
    progress(stances_left); print("Finshed Left separation")
    stances_right = stances_data[stances_data['p'] > .66].drop(['timestamp', 'p'], axis=1).persist(retries=100)
    progress(stances_right); print("Finshed Right separation")

    # write edgelists
    stances_left.to_csv(os.path.join(WORKING_DIR, 'url_classified_edgelists', 'pro_biden_retweet_edges.csv'), single_file=True)
    stances_right.to_csv(os.path.join(WORKING_DIR, 'url_classified_edgelists', 'pro_trump_retweet_edges.csv'), single_file=True)
    print("Finished writing stance edgelists!")
    return


def dask_str_col_to_int(data, col):
    to_numeric = lambda df: pd.to_numeric(df, errors='coerce')
    data[col] = data[col].map_partitions(to_numeric)
    data = data.dropna()
    data[col] = data[col].astype(np.int64)
    return data


def tweet_to_leaning():
    tweet_leanings = {}
    # if we have not already created our tweet to leaning intermediary file, create it now.
    if not os.path.isfile(os.path.join(WORKING_DIR, 'intermediaries', 'tweet_classes.csv')):
        print("Couldn't load intermediary data. Creating intermediary files.")
        tweet_to_class()

    # load our tweets to leanings

    leanings_path = '/home/pub/hernan/Election_2020/classified_tweets.csv'
    urls_data = dd.read_csv(leanings_path, delimiter=',',
                            usecols=['tweet_id', 'bias'],
                            dtype={'tweet_id': np.int64, 'bias': str}
                            ).rename(columns={'tweet_id': 'id', 'bias': 'leaning'}).set_index('id')
    classes = urls_data['leaning'].unique()



    # loop over each unique leaning type and create a dict of leaning to tweets
    for leaning in classes:
        print("Loading {} tweets".format(leaning))
        tweet_leanings[leaning] = urls_data[urls_data['leaning'] == leaning].persist(retries=1000)
        progress(tweet_leanings[leaning])

    return tweet_leanings


def read_edgeslist(path, columns=('auth_id', 'infl_id')):
    edges = dd.read_csv(path, delimiter=',',
                        usecols=columns,
                        dtype={col: np.int64 for col in columns}
                        )
    return edges


def polarization_filter():
    """
    This filter aims at removing any noise in our networks generated by the increase
    in polarization of twitter. We have found a number of instances (with only the smallest searching)
    of users getting high influence in a network where most of the influence was generated through
    conflict, i.e. some argument thread, or bashing on some user by @'ing him, etc.
    This informatino is useful when we want to learn about the change in behavior towards news outlets,
    or polarization of political opinion and increased tension, but when we want to analyze which figures
    in a network propagate the most information/misinformation to a particular political base, we need
    to remove this conflict noise, since it doesn't represent the propagation of information.

    To remove this noise, we can use the stance data, which represents the political leanings of a
    user on a given day. (as represented by a value between 0-1 where 1 is pro trump and 0 is pro biden).
    By performing an inner join on edges that are pro biden and edges that are also left leaning (by url classification)
    we can remove a good amount of users who write right leaning posts but then reply to biden about how he is a
    dumb dumb or somthing.
    :return:
    """
    # load our stance data
    stances = ['left', 'right']
    stance_to_leanings = {'left': ['Left news', 'Extreme bias left', 'Left leaning news'],
                          'right': ['Right news', 'Extreme bias right', 'Right leaning news']}
    columns = ['id', 'infl_id', 'auth_id']
    for stance in stances:
        s_path = '/home/crossb/working/old_merged_tweets/{}_edges.csv'.format(stance)
        stance_edges = read_edgeslist(s_path, columns=columns).set_index('id')

        # get leaning edges
        for leaning in stance_to_leanings[stance]:
            l_path = os.path.join(WORKING_DIR, 'url_classified_edgelists', '{}_edges.csv'.format(leaning))
            leaning_edges = read_edgeslist(l_path, columns=columns).set_index('id')

            # inner merge the two edgelists
            merged_edges = leaning_edges.merge(stance_edges, on=columns)

            # write to file
            merged_edges.to_csv(
                os.path.join(WORKING_DIR, 'url_classified_edgelists', '{}_stance_filtered_edges.csv'.format(leaning)),
                single_file=True)

    return


def apply_tweet_leanings(tweets, tweet_leaning_filters=None):
    tweet_leanings = tweet_to_leaning()

    for leaning, data in tweet_leanings.items():
        print("Begin {} edgelist".format(leaning))
        stime = time.time()
        #class_data = urls_data[urls_data['leaning'] == leaning].persist(retries=1000)
        # class_data = class_data
        #progress(class_data)
        print("Split urls by leaning")
        # merge
        tweet_leanings[leaning] = tweets.merge(data, left_index=True, right_index=True).persist(retries=1000)
        progress(tweet_leanings[leaning])

        # filter out biden / trump from right leaning networks and left leaning networks respectively
        #left_leanings = ['Left leaning news', 'Extreme bias left', 'Left news']
        #right_leanings = ['Right leaning news', 'Extreme bias right', 'Right news']
        #if leaning in left_leanings:
        #    tweet_leanings[leaning] = tweet_leanings[leaning][tweet_leanings[leaning]['auth_id'] != 25073877]
        #elif leaning in right_leanings:
        #    tweet_leanings[leaning] = tweet_leanings[leaning][tweet_leanings[leaning]['auth_id'] != 939091]

    return tweet_leanings


def assign_edge_classes(tweets, name='', write=True):
    print("Begin assigning edge classes")
    bias_to_filename = {'Center news': 'center', 'Fake news': 'fake', 'Extreme bias right': 'right_extreme',
                        'Extreme bias left': 'left_extreme', 'Left leaning news': 'left_leaning',
                        'Right leaning news': 'right_leaning', 'Left news': 'left', 'Right news': 'right'}
    stime_total = time.time()
    #write_dir = '/home/crossb/working/url_classified_edgelists'
    write_dir = '/home/crossb/working/url_classified_edgelists/retweet_edges_fixed_missing_01302020'

    # read the tweet classes
    #leanings_path = os.path.join(WORKING_DIR, 'intermediaries', 'tweet_classes.csv')
    #leanings_path = '/home/pub/hernan/Election_2020/classified_tweets.csv'
    leanings_path = '/home/pub/hernan/Election_2020/classified_data_with_double_links_v2.csv'
    urls_data = dd.read_csv(leanings_path, delimiter=',',
                            usecols=['tweet_id', 'bias'],#usecols=['id', 'leaning'],
                            dtype={'tweet_id': np.int64, 'bias': str}
                            ).rename(columns={'tweet_id': 'id', 'bias': 'leaning'}).set_index('id')
    classes = urls_data['leaning'].unique()
    #print("Classes: {}".format(classes))
    for edge_class in classes:

        print("Begin {} edgelist".format(edge_class))
        stime = time.time()
        class_data = urls_data[urls_data['leaning'] == edge_class].persist(retries=1000)
        #class_data = class_data
        progress(class_data)
        print("Split urls by leaning")
        # merge
        class_data = tweets.merge(class_data, left_index=True, right_index=True).persist(retries=1000)
        progress(class_data)

        if write:
            print("Merged Tweets to urls")
            drop_cols = ['timestamp', 'leaning']#, 'p']
            class_data = class_data.drop(drop_cols, axis=1)

            print("Writing edgelist to {}".format(write_dir))
            class_data.to_csv(os.path.join(write_dir, '{}_{}_edges.csv'.format(bias_to_filename[edge_class], name)),
                              single_file=True)
            #class_data.to_csv(os.path.join(write_dir, '{}_edges.csv'.format(edge_class)), single_file=True, index=False)
            print("{} edgelist elapsed time {} seconds".format(edge_class, time.time() - stime))

    print("Assign edge classes elapsed time {} seconds".format(time.time() - stime_total))
    return class_data


# Add methods for the creation of twitter
def tweet_to_class():
    print("Begin correlating tweets to stances!")
    stime = time.time()
    # READ THE URLS DATA
    urls_data = dd.read_csv(os.path.join(URLS_DIR, '*.csv'), delimiter=',',
                               usecols=['tweet_id', 'user_id', 'expanded_url'],
                               dtype={'tweet_id': str, 'user_id': str, 'expanded_url': str}
                            ).rename(columns={'tweet_id': 'id', 'expanded_url': 'url'})
    urls_data = dask_str_col_to_int(urls_data, 'id')

    # READ CLASSIFIED LINKS
    classifications = dd.read_csv(os.path.join(CLASSIFIED_URLS_DIR, '*.tsv'), delimiter='\t',
                                  usecols=['url', 'leaning'],
                                  dtype={'url': str, 'leaning': str}, quotechar='"')

    classifications = urls_data.merge(classifications, on='url')
    classifications = classifications.dropna().drop_duplicates().persist(retries=1000)
    #classifications = classifications.persist(retries=1000)
    progress(classifications)
    print("Dropped duplicates!")

    classifications = classifications.drop(['url'], axis=1)

    classifications.to_csv(os.path.join(WORKING_DIR, 'intermediaries', 'tweet_classes.csv'),
                           index=False, single_file=True)
    print("Stance correlation took {} seconds".format(time.time() - stime))
    return


def fake_news_retweet_networks(tweets):
    write_dir = '/home/crossb/working/url_classified_edgelists'
    tweet_to_fake_news = fake_news_urls()
    for edge_class, class_data in tweet_to_fake_news.items():
        name = edge_class + 'fake news'
        print("Begin {} edgelist".format(name))
        stime = time.time()
        class_data.set_index('id')
        class_data.persist(retries=1000)
        progress(class_data)
        print("Split urls by leaning")
        # merge
        #class_data = tweets.merge(class_data, on=['id']).persist(retries=1000)
        class_data = tweets.merge(class_data, left_index=True, right_index=True).persist(retries=1000)
        progress(class_data)
        print("Merged Tweets to urls")
        class_data = class_data.drop(['timestamp', 'leaning', 'type'], axis=1)

        #progress(class_data)
        class_data.to_csv(os.path.join(write_dir, '{}_edges.csv'.format(name)),
                          single_file=True)
        #class_data.to_csv(os.path.join(write_dir, '{}_edges.csv'.format(edge_class)), single_file=True, index=False)
        print("{} edgelist elapsed time {} seconds".format(edge_class, time.time() - stime))
    return


def fake_news_urls():
    print("Begin fake news urls wrangling")
    stime = time.time()
    # READ THE URLS DATA
    urls_data = dd.read_csv(os.path.join(URLS_DIR, '*.csv'), delimiter=',',
                            usecols=['tweet_id', 'expanded_url'],
                            dtype={'tweet_id': str, 'expanded_url': str}
                            ).rename(columns={'tweet_id': 'id', 'expanded_url': 'url'})
    urls_data = dask_str_col_to_int(urls_data, 'id')

    # READ CLASSIFIED LINKS
    classifications = dd.read_csv(os.path.join(CLASSIFIED_URLS_DIR, '*.tsv'), delimiter='\t',
                                  usecols=['url', 'type', 'leaning'],
                                  dtype={'url': str, 'type': str, 'leaning': str}, quotechar='"')

    classifications = urls_data.merge(classifications, on='url')
    classifications = classifications.dropna().drop_duplicates()
    classifications = classifications.persist(retries=1000)
    progress(classifications)
    classifications = classifications.drop(['url'], axis=1)

    print("Stance correlation took {} seconds".format(time.time() - stime))

    for leaning in ['left', 'right', 'all']:
        if leaning == 'left':
            filters = ['left', 'Left']
        elif leaning == 'right':
            filters = ['right', 'Right']
        else:
            filters = []

    # get our different desired fake news classes
    universal_fake_news = classifications[classifications['type'] == 'unreliable']
    left_fake = universal_fake_news[(universal_fake_news['leaning'] == 'Left') |
                                    (universal_fake_news['leaning'] == 'Left-Center') |
                                    (universal_fake_news['leaning'] == 'extreme left')]
    right_fake = universal_fake_news[(universal_fake_news['leaning'] == 'Right') |
                                    (universal_fake_news['leaning'] == 'Right-Center') |
                                    (universal_fake_news['leaning'] == 'extreme right')]

    fake_news_by_stance = {'universal': universal_fake_news, 'left': left_fake, 'right': right_fake}
    return fake_news_by_stance


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


def trump_biden_deficient_networks():
    """
    Remove trump from our left leaning networks and biden from our right leaning networks.
    :return:
    """
    print("Removing trump and biden")
    write_dir = '/home/crossb/working/url_classified_edgelists'
    for filename in ['Left leaning news', 'Left news', 'Extreme bias left']:
        path = os.path.join(write_dir, "{}_edges.csv".format(filename))
        edges = dd.read_csv(path, delimiter=',',
                            usecols=['id', 'auth_id', 'infl_id'],
                            dtype={'id': np.int64, 'auth_id': np.int64, 'infl_id': np.int64}
                            ).set_index('id')
        edges = edges[edges['infl_id'] != 25073877].persist(retries=100)
        progress(edges)
        print("Filtered trump out of {}".format(filename))
        edges.to_csv(os.path.join(write_dir, '{}_edges_less_trump.csv'.format(filename)),
                          single_file=True)
    for filename in ['Right leaning news', 'Right news', 'Extreme bias right']:
        path = os.path.join(write_dir, "{}_edges.csv".format(filename))
        edges = dd.read_csv(path, delimiter=',',
                            usecols=['id', 'auth_id', 'infl_id'],
                            dtype={'id': np.int64, 'auth_id': np.int64, 'infl_id': np.int64}
                            ).set_index('id')
        edges = edges[edges['infl_id'] != 939091].persist(retries=100)
        progress(edges)
        print("Filtered biden out of {}".format(filename))
        edges.to_csv(os.path.join(write_dir, '{}_edges_less_biden.csv'.format(filename)),
                     single_file=True)

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


def DEBUG():
    import glob

    # load tweet_ids from tweets, retweets, and classifications
    #glob_0_9 = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'tweets', '20200[6-9]*.csv'))
    #glob_tens = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'tweets', '2020[1][0-1]*.csv'))
    tweets_path = os.path.join(Election_2020_dir, 'tweets', '*.csv')

    glob_0_9 = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'retweets', '20200[6-9]*.csv'))
    glob_tens = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'retweets', '2020[1][0-1]*.csv'))
    retweets_path = glob_0_9 + glob_tens#os.path.join(Election_2020_dir, 'retweets', '*.csv')

    classifications_path = os.path.join("/home/pub/hernan/Election_2020/classified_data_with_double_links_v2.csv")

    glob_0_9 = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'quotes', '20200[6-9]*.csv'))
    glob_tens = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'quotes', '2020[1][0-1]*.csv'))
    quotes_path = glob_0_9 + glob_tens#os.path.join(Election_2020_dir, 'quotes', '*.csv')

    glob_0_9 = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'replies', '20200[6-9]*.csv'))
    glob_tens = glob.glob(os.path.join('/home/pub/hernan/Election_2020/joined_output', 'replies', '2020[1][0-1]*.csv'))
    replies_path = glob_0_9 + glob_tens#os.path.join(Election_2020_dir, 'replies', '*.csv')
    #mentions_path = os.path.join(Election_2020_dir, 'user_mentions', '*.csv')
    #classifications_path = os.path.join("/home/pub/hernan/Election_2020/classified_tweets.csv")

    tweets = dd.read_csv(tweets_path, delimiter=',', usecols=['tweet_id', 'created_at'],
                         dtype={'tweet_id': str, 'created_at': str}).rename(columns={'tweet_id': 'id'})
    tweets = dask_str_col_to_int(tweets, 'id').persist(retries=100)
    tweets = tweets.dropna()
    tweets['id'] = tweets['id'].astype(np.int64)
    tweets['created_at'] = dd.to_datetime(tweets['created_at'], format='%Y-%m-%d %H:%M:%S')
    filter_dates = lambda df: df[(df['created_at'] > start_date)]# & (df['created_at'] < stop_date)]
    tweets = tweets.map_partitions(filter_dates)
    tweets = tweets.persist()
    progress(tweets);
    print("Tweets: Filtered tweets on datetime")


    #retweets = dd.read_csv(retweets_path, delimiter=',', usecols=['id'], dtype={'id': str})
    #retweets = dask_str_col_to_int(retweets, 'id').persist(retries=100)
    #progress(retweets); print('Loaded retweets')
#
    #quotes = dd.read_csv(quotes_path, delimiter=',', usecols=['id'], dtype={'id': str})
    #quotes = dask_str_col_to_int(quotes, 'id').persist(retries=100)
    #progress(quotes); print('Loaded quotes')
#
    #replies = dd.read_csv(replies_path, delimiter=',', usecols=['id'], dtype={'id': str})
    #replies = dask_str_col_to_int(replies, 'id').persist(retries=100)
    #progress(replies); print('Loaded replies')
#
    #user_mentions = dd.read_csv(mentions_path, delimiter=',', usecols=['tweet_id'], dtype={'tweet_id': str}).rename(columns={'tweet_id': 'id'})
    #user_mentions = dask_str_col_to_int(user_mentions, 'id').persist(retries=100)
    #progress(user_mentions); print('Loaded user_mentions')

    classifications = dd.read_csv(classifications_path, delimiter=',', usecols=['tweet_id'],
                                  dtype={'tweet_id': str}).rename(columns={'tweet_id': 'id'})
    classifications = dask_str_col_to_int(classifications, 'id').persist(retries=100)
    progress(classifications); print('Loaded classifications')

    tweets = tweets.compute()
    print("Earliest Datetime is {}".format(tweets.created_at.min()))
    print("Latest Datetime is {}".format(tweets.created_at.max()))
    if True:
        return
    #retweets = retweets.compute()
    #quotes = quotes.compute()
    #replies = replies.compute()
    #user_mentions = user_mentions.compute()
    classifications = classifications.compute()

    unique_tweets = set(tweets.id.unique())#np.array(tweets.id.unique())
    #unique_retweets = set(retweets.id.unique())
    #unique_quotes = set(quotes.id.unique())
    #unique_replies = set(replies.id.unique())
    #unique_user_mentions = np.array(user_mentions.id.unique())
    #unique_retweet_ids = np.array(retweets.retweet_id.unique())
    unique_classifications = set(classifications.id.unique())#np.array(classifications.id.unique())

    print("Unique tweet ids:", len(unique_tweets))
    #print("Unique retweet ids:", len(unique_retweets))
    #print("Unique quotes ids:", len(unique_quotes))
    #print("Unique replies ids:", len(unique_replies))
    #print("Unique user mentions ids:", len(unique_user_mentions))
    print("Unique tweet id classifications:", len(unique_classifications))


    same_tweets_classifications = unique_classifications.intersection(unique_tweets)#np.intersect1d(unique_tweets, unique_classifications)
    print("Tweets.id intersection Classifications.tweet_id", len(same_tweets_classifications))

    difference = unique_classifications.difference(unique_tweets)
    #df = pd.DataFrame()
    #df['missing_tweet_id'] = np.array(list(difference))
    #pdb.set_trace()
    #df.to_csv('missing_tweet_ids.csv', index=False)
    same_tweets_retweets = unique_retweets.intersection(unique_tweets)
    print("Tweets.id intersection Retweets.id", len(same_tweets_retweets))
    #same_tweet_id_retweet_ids = np.intersect1d(unique_tweets, unique_retweet_ids)
    #print("Tweets.id intersection Retweets.retweet_id", len(same_tweet_id_retweet_ids))
    same_retweets_classifications = unique_classifications.intersection(unique_retweets)
    print("Retweets.id intersection Classifications.tweet_id", len(same_retweets_classifications))

    same_quotes_classifications = unique_classifications.intersection(unique_quotes)
    print("Quotes.id intersection Classifications.tweet_id", len(same_quotes_classifications))
    same_replies_classifications = unique_classifications.intersection(unique_replies)
    print("Replies.id intersection Classifications.tweet_id", len(same_replies_classifications))
    #same_user_mentions_classifications = np.intersect1d(unique_user_mentions, unique_classifications)
    #print("User Mentions.id intersection Classifications.tweet_id", len(same_user_mentions_classifications))

    same_all = unique_classifications.intersection(same_tweets_retweets)
    print("Tweets.id.intersect(retweets.id).intersect(classifications.tweet_id)", len(same_all))


    #path = os.path.join(WORKING_DIR, 'url_classified_edgelists', 'Left leaning news_edges_less_trump.csv')
    #edges = dd.read_csv(path, delimiter=',',
    #                    usecols=['id', 'auth_id', 'infl_id'],
    #                    dtype={'id': np.int64, 'auth_id': np.int64, 'infl_id': np.int64}
    #                    )
#
    #edges = edges.persist(retries=1000)
    #progress(edges)
    #edges = edges.compute()
    #pdb.set_trace()

    return


def merge_tweets_and_stances():
    stance_data = dask_filter_stance()
    print("Compute stance data")
    stime = time.time()
    #stance_data = stance_data.compute()
    stance_data = stance_data.repartition(npartitions=150).persist(retries=100)
    progress(stance_data)
    print("Computation took: {} seconds".format(time.time() - stime))
    #stime = time.time()
    #stance_data = stance_data.set_index(['auth_id', 'timestamp'])
    #print("Multi-Index time taken: {}".format(time.time() - stime))
    tweet_dir = os.path.join(Election_2020_dir, 'tweets')
    tweet_files = [f for f in os.listdir(tweet_dir) if os.path.isfile(os.path.join(tweet_dir, f))]
    # for each tweet file
    for file in tweet_files:
        ftime = time.time()
        print("Starting {}".format(file))
        tweet_data = dd.read_csv(os.path.join(tweet_dir, file), sep=',', usecols=['tweet_id', 'created_at', 'user_id'],
                                 dtype={'tweet_id': str, 'created_at': str, 'user_id': str}
                                 ).rename(columns={'tweet_id': 'id', 'created_at': 'timestamp', 'user_id': 'auth_id'})

        tweet_data = tweet_data[tweet_data['id'] != 'id']  # %H:%M:%S')
        tweet_data = dask_str_col_to_int(tweet_data, 'auth_id')
        tweet_data = dask_str_col_to_int(tweet_data, 'id')

        # remove duplicates

        tweet_data = tweet_data.drop_duplicates(subset='id').persist(retries=100)
        progress(tweet_data)
        print("Tweets: Dropped duplicates")
        tweet_data = tweet_data.repartition(npartitions=150).persist(retries=100)

        full_timestamp = tweet_data[['id', 'timestamp']].set_index('id')
        to_date = lambda df: df['timestamp'].str.slice(stop=10)
        tweet_data['timestamp'] = tweet_data.map_partitions(to_date)
        tweet_data['timestamp'] = dd.to_datetime(tweet_data['timestamp'], format='%Y-%m-%d')

        tweet_data = tweet_data.persist(retries=100)
        progress(tweet_data)
        full_timestamp = full_timestamp.persist(retries=100)
        progress(full_timestamp)
        #tweet_data = tweet_data.compute()
        #tweet_data = tweet_data.set_index(['auth_id', 'timestamp'])
        #full_timestamp = full_timestamp.compute()
        print("Tweets loaded and formatted")
        #out_data = tweet_data.merge(stance_data, left_index=True, right_index=True, how='left').set_index('id')
        tweet_data = tweet_data.merge(stance_data, on=['auth_id', 'timestamp'], how='left').persist(retries=100)
        progress(tweet_data)
        print("Merged tweets and stance")
        tweet_data = tweet_data.set_index('id').persist(retries=100)
        #.set_index('id')#.persist(retries=1000)
        progress(tweet_data)
        print("Set index")
        tweet_data = tweet_data.drop(['timestamp'], axis=1)

        tweet_data = tweet_data.merge(full_timestamp, left_index=True, right_index=True).persist(retries=100)
        progress(tweet_data)
        print("Dropped ")
        #tweet_data['timestamp'] = full_timestamp

        print("Tweets and stances merged")
        tweet_data.to_csv(os.path.join(WORKING_DIR, 'stance_merged_tweets', 'corrected_stance_{}'.format(file)),
                          single_file=True)
        print("Time taken:", time.time() - ftime)

    return


def fix_tweet_stance_timestamps():
    tweet_dir = os.path.join(Election_2020_dir, 'tweets')
    #tweets = dd.read_csv(os.path.join(tweet_dir, '*.csv'), sep=',', usecols=['id', 'created_at'],
    #                             dtype={'id': str, 'created_at': str}
    #                             )#.set_index('id').persist(retries=100)
    #tweets = tweets[tweets['id'] != 'id']
    #tweets = dask_str_col_to_int(tweets, 'id').set_index('id').persist(retries=100)
    #progress(tweets)
    #print("Loaded tweets")

    stance_tweet_files = [f for f in os.listdir(WORKING_DIR) if os.path.isfile(os.path.join(WORKING_DIR, f))]
    for file in stance_tweet_files:
        stime = time.time()
        if not file.endswith('tweets.csv'):
            continue

        # load corresponding tweet file
        tweet_file = file[7:]
        tweets = dd.read_csv(os.path.join(tweet_dir, tweet_file), sep=',', usecols=['id', 'created_at'],
                             dtype={'id': str, 'created_at': str}
                             )  # .set_index('id').persist(retries=100)
        tweets = tweets[tweets['id'] != 'id']
        tweets = dask_str_col_to_int(tweets, 'id').set_index('id').persist(retries=100)
        progress(tweets)
        print("Loaded tweets")


        print("Staring file: {}".format(file))
        stance_tweets = dd.read_csv(os.path.join(WORKING_DIR, file), sep=',', usecols=['id', 'auth_id', 'p'],
                                    dtype={'id': np.int64, 'auth_id': np.int64, 'p': np.float64}
                                    ).set_index('id').persist(retries=100)
        progress(stance_tweets)

        #stance_tweets = stance_tweets.compute()
        #tweets = tweets.compute()
#
        #tweets_ids = np.array(tweets.index)
        #stance_tweets_ids = np.array(stance_tweets.index)
        #diff = np.setdiff1d(stance_tweets_ids, tweets_ids)
        #
        #tweets_nodup = tweets.drop_duplicates().persist(retries=100)
        tweets_nodups = tweets.reset_index().drop_duplicates(subset=['id']).set_index('id').persist(retries=100)
        #progress(tweets_nodup)
        print("Loaded stance tweets")
        #stance_tweets = tweets.merge(stance_tweets, right_index=True, left_index=True).persist(retries=100)
        stance_tweets = stance_tweets.merge(tweets_nodups, left_index=True, right_index=True, how='inner').persist(retries=100)
        #stance_tweets = stance_tweets.reset_index()
        #tweets = tweets.reset_index()
        #m2 = stance_tweets.merge(tweets, on='id',).persist(retries=100)
        #progress(m1)
        progress(stance_tweets)

        #pdb.set_trace()
        print("Merged proper timestamp")
        stance_tweets.to_csv(os.path.join(WORKING_DIR, 'correct_stance_{}'.format(file)), single_file=True)#, index=False)
        print("Time taken", time.time() - stime)
    return


def load_retweet_intermediaries():
    path = os.path.join(WORKING_DIR, 'intermediaries', 'retweets')#'test')
    print("Loading intermediaries from {}".format(path))
    data = dd.read_csv(os.path.join(path, '*.csv'), delimiter=',',
                               dtype={'id': np.int64, 'timestamp': str, 'auth_id': np.int64,
                                      #'user_id': np.int64,
                                      #'p': np.float64,
                                      'infl_id': np.int64}).set_index('id').persist(retries=100)
    progress(data); print("Finished loading stance intermediaries")
    return data


def gather_edges():
    print("Begin gathering edge data")
    #dask_load_tweets(os.path.join(Election_2020_dir, 'tweets'))
    # Step 1: Process/Load our twitter data and correlate that data to our twitter responses
    #for dir in [os.path.join(Election_2020_dir, 'retweets'),
    #            os.path.join(Election_2020_dir, 'replies'),
    #            os.path.join(Election_2020_dir, 'quotes')]:
    #    correlate(dir)
    # user mentions is formatted without a corresponding response retweet/reply, etc. id, so we will process
    # differently
    #correlate(os.path.join(Election_2020_dir, 'user_mentions'), remove_duplicates=False)

    #retweet network
    #correlate(os.path.join(Election_2020_dir, 'retweets'), remove_duplicates=True)

    # Step 2: Concatenate all the edges from step 1 into one edgelist
    #edges = concat_edges(type='retweet')

    # Optional step 3: Classify edges by stance
    # write to file
    #dask_filter_stance()

    # step 4: Classify edges by political leaning
    ## URLS ##
    #tweet_to_class()


    # HERE BE THE PLACE TO REGENERATE =======================================================
    #edges = load_retweet_intermediaries()
    #assign_edge_classes(edges, name='retweet')

    #trump_biden_deficient_networks()
    ## STANCES ##
    # separate by stances
    # dask_assign_stance(edges)

    #polarization_filter()

    DEBUG()
    #merge_tweets_and_stances()
    #fix_tweet_stance_timestamps()
    #gather_retweet_edges_v2(os.path.join(WORKING_DIR, 'stance_merged_tweets'), os.path.join(Election_2020_dir, 'retweets'))
    #write_stance_edgelists(os.path.join(WORKING_DIR, 'stance_merged_retweets'))

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


def concat_edges(type='', write=False):
    stime = time.time()
    columns = ['id', 'infl_id', 'timestamp', 'auth_id']
    data = dd.read_csv(os.path.join(WORKING_DIR, 'intermediaries', 'merged_tweet_to_{}*.csv'.format(type)),
                       delimiter=',', usecols=columns,
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


def correlate_ids():
    all_time = time.time()
    tweet_ids = []
    retweet_ids = []
    tweet_dir = os.path.join(Election_2020_dir, 'tweets')
    retweet_dir = os.path.join(Election_2020_dir, 'retweets')
    tweet_files = [f for f in os.listdir(tweet_dir) if os.path.isfile(os.path.join(tweet_dir, f))]
    retweet_files = [f for f in os.listdir(retweet_dir) if os.path.isfile(os.path.join(retweet_dir, f))]

    for tweet_file in tweet_files:
        ftime = time.time()
        print("Starting {}".format(tweet_file))
        with open(os.path.join(tweet_dir, tweet_file), 'r') as tweet_csv:
            for line in tweet_csv.readlines():
                row = [x for x in line.strip().split(',')]
                try:
                    created_at = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
                except ValueError as e:
                    # print("Format mismatch with following data: {},{},{}".format(*row))
                    continue
                if created_at >= start_date and created_at < stop_date:
                    retweet_ids.append(int(row[0]))
        print("Time taken", time.time() - ftime)
    print("Done reading tweets")
    tweet_ids = np.array(tweet_ids)
    print("Starting retweets")
    for retweet_file in retweet_files:
        ftime = time.time()
        print("Starting {}".format(retweet_file))
        with open(os.path.join(retweet_dir, retweet_file), 'r') as retweet_csv:
            for line in retweet_csv.readlines():
                row = [x for x in line.strip().split(',')]
                retweet_ids.append(int(row[0]))
        print("Time taken", time.time() - ftime)

    print("Finished reading retweets")
    retweet_ids = np.array(retweet_ids)

    print("Starting correlation")
    inter_time = time.time()
    in_both = np.intersect1d(tweet_ids, retweet_ids)
    print("Intersect took {} seconds".format(time.time() - inter_time))
    print("Time taken total =", time.time() - all_time)
    pdb.set_trace()
    return in_both


def correlate_pandas():
    all_time = time.time()
    tweet_ids = []
    retweet_ids = []
    tweet_dir = os.path.join(Election_2020_dir, 'tweets')
    retweet_dir = os.path.join(Election_2020_dir, 'retweets')
    tweet_files = [f for f in os.listdir(tweet_dir) if os.path.isfile(os.path.join(tweet_dir, f))]
    retweet_files = [f for f in os.listdir(retweet_dir) if os.path.isfile(os.path.join(retweet_dir, f))]
    tweet_df = None
    for file in tweet_files:
        ftime = time.time()
        print("Starting {}".format(file))
        if tweet_df is None:
            tweet_df = pd.read_csv(os.path.join(tweet_dir, file),
                                   sep=',', usecols=['id', 'created_at'])
        else:
            tweet_df = tweet_df.append(pd.read_csv(os.path.join(tweet_dir, file),
                                        sep=',', usecols=['id', 'created_at']))
        print("num_retweet_rows:", len(tweet_df))
        print("Time taken", time.time() - ftime)
    tweet_df.id = pd.to_numeric(tweet_df.id, errors='coerce').dropna().astype(np.int64)
    tweet_df.created_at = pd.to_datetime(tweet_df.created_at, format='%Y-%m-%d %H:%M:%S', errors='coerce')#.dropna()
    tweet_df = tweet_df[(tweet_df.created_at > start_date)&(tweet_df.created_at < stop_date)]
    tweet_ids = np.array(tweet_df.id).astype(np.int64)

    df = None
    for file in retweet_files:
        ftime = time.time()
        print("Starting {}".format(file))
        if df is None:
            df = pd.read_csv(os.path.join(retweet_dir, file),
                                   sep=',', usecols=['id'])
        else:
            df = df.append(pd.read_csv(os.path.join(retweet_dir, file),
                                        sep=',', usecols=['id']))
        print("num_retweet_rows:", len(df))
        #print("min retweet id:", np.min(df.id), "max retweet id:", np.max(df.id))
        print("Time taken", time.time() - ftime)
    df.id = pd.to_numeric(df.id, errors='coerce').dropna().astype(np.int64)
    retweet_ids = np.array(df.id).astype(np.int64)

    print("Starting correlation")
    inter_time = time.time()
    in_both = np.intersect1d(tweet_ids, retweet_ids)
    print("Intersect took {} seconds".format(time.time() - inter_time))

    print("Time taken {}".format(time.time() - all_time))
    pdb.set_trace()
    return


def main():
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=1,
                           scheduler_port=0, dashboard_address=None, memory_limit='256GB')
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

