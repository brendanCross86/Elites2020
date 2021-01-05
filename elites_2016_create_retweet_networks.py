import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from urllib.parse import urlparse
from urllib.parse import unquote
#from PlotUtils import addDatetimeLabels, add_vspans
from GraphUtils import buildGraphSqlite, buildGraphCSV
from datetime import datetime
from collections import Counter

import time
import pickle
import graph_tool.all as gt
import pdb
#from TwStats import chunks

#from PlotUtils import compute_CCDF

#save_dir = '../data/urls/revisions/'
save_dir = '../data/test2016'

#raise Exception

#%% load user and tweet list
#tweet_db_file1 = '../databases_ssd/complete_trump_vs_hillary_db.sqlite'
#tweet_db_file2 = '../databases_ssd/complete_trump_vs_hillary_sep-nov_db.sqlite'
#urls_db_file = '../databases_ssd/urls_db.sqlite'
tweet_db_file1 = '/home/pub/hernan/Election_2016/complete_trump_vs_hillary_db.sqlite'
tweet_db_file2 = '/home/pub/hernan/Election_2016/complete_trump_vs_hillary_sep-nov_db.sqlite'
media_types = ['left', 'right']#, 'central']
media_type_to_query = {
    'left': "SELECT tweet_id FROM class_proba WHERE p_pro_hillary_anti_trump > .66",
    'right': "SELECT tweet_id FROM class_proba WHERE p_pro_hillary_anti_trump < .33"
}

t00 = time.time()
t0 = time.time()
#
start_date = datetime(2016, 6, 1)
stop_date = datetime(2016, 11, 9)
edges_db_file = dict()

# get edges list

# set tmp dir
os.environ['SQLITE_TMPDIR'] = '/home/crossb'
# BELOW FOR LOOP IS FOR 2016 DATA IN SQLITE DB
for tweet_db_file in [tweet_db_file1, tweet_db_file2]:
    print(tweet_db_file)

    edges_db_file[tweet_db_file] = dict()
    with sqlite3.connect(tweet_db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
        c = conn.cursor()

        for media_type, additional_select in media_type_to_query.items():
            edges_db_file[tweet_db_file][media_type] = buildGraphSqlite(
                conn,
                graph_type='retweet',
                start_date=start_date,
                stop_date=stop_date,
                additional_sql_select_statement=additional_select,
                    graph_lib='edge_list')

        print(time.time() - t0)

    print(time.time() - t0)

# save
with open(os.path.join(save_dir,'edge_lists.pickle'), 'wb') as fopen:
    pickle.dump(edges_db_file, fopen)

# save
# with open(os.path.join(save_dir,'edge_lists_not_aggr_not_shareblue.pickle'), 'wb') as fopen:
# pickle.dump(edges_db_file, fopen)

#
# edges_db_file = dict()
# edges_db_file[tweet_db_file1] = edge_list1[tweet_db_file1]
# edges_db_file[tweet_db_file2] = edge_list2[tweet_db_file2]
# %% add source to edges


#def get_tweet_source(edge_list, tweet_db_file):
#    tweet_id_list = edge_list[:, 2].tolist()
#
#    tweet_source_list = []
#    with sqlite3.connect(tweet_db_file,
#                         detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
#        c = conn.cursor()
#        for chunk in chunks(tweet_id_list, 999):
#            sql_add_source = """SELECT tweet_id, source_content_id FROM tweet
#                        WHERE tweet_id IN ({tids})""" \
#                .format(tids=','.join(['?'] * len(chunk)))
#
#            c.execute(sql_add_source, chunk)
#
#            tweet_source_list.extend([tc for tc in c.fetchall()])
#
#    assert len(tweet_id_list) == len(tweet_source_list)
#
#    return tweet_source_list


#t0 = time.time()
#
#source_db_file = dict()
#
#for tweet_db_file in edges_db_file.keys():
#    source_db_file[tweet_db_file] = get_tweet_source(edges_db_file[tweet_db_file],
#                                                     tweet_db_file)
#    print(time.time() - t0)

# %% get source name from mapping

#with sqlite3.connect(tweet_db_file1,
#                     detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
#    c = conn.cursor()
#    c.execute("SELECT * FROM source_content")
#
#    source_content_map1 = dict((i, s) for i, s in c.fetchall())
#
#with sqlite3.connect(tweet_db_file2,
#                     detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
#    c = conn.cursor()
#    c.execute("SELECT * FROM source_content")
#
#    source_content_map2 = dict((i, s) for i, s in c.fetchall())
#
## %% read mapping from db
#with sqlite3.connect(urls_db_file,
#                     detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES) as conn:
#    c = conn.cursor()
#    c.execute("SELECT id, source_content FROM source_content_to_id_map")
#
#    # source name to source id
#    source_global_map = dict((s, i) for i, s in c.fetchall())
#
#    # source id to souce name
#    source_global_map_id2sn = dict((i, s) for s, i in source_global_map.items())
## %%    # add source names to edges
#
#official_twitter_clients_ids = [source_global_map[tc] for tc in official_twitter_clients]
#
#for tweet_db_file in edges_db_file.keys():
#
#    for media_type, edge_list in edges_db_file[tweet_db_file].items():
#
#        print(tweet_db_file)
#        print(media_type)
#
#        tweet_sources = source_db_file[tweet_db_file][media_type]
#
#        if tweet_db_file == tweet_db_file1:
#            source_content_map = source_content_map1
#        elif tweet_db_file == tweet_db_file2:
#            source_content_map = source_content_map2
#        else:
#            raise Exception
#
#        edges_db_file[tweet_db_file][media_type] = [(infid, authid, tid,
#                                                     source_global_map[source_content_map[tweet_sources[i][1]]]) \
#                                                    for i, (infid, authid, tid) in enumerate(edge_list)]


# %% build graphs

def build_graph_from_edges(edge_list, graph_name):
    G = gt.Graph(directed=True)
    G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
    G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')
    G.edge_properties['source_id'] = G.new_edge_property('int64_t')
    G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id, G.ep.source_id])

    G.gp['name'] = G.new_graph_property('string', graph_name)

    return G


# compute ratio of tweets from official clients
def get_vertex_official_client_ratio(v, source_property_map, direction='in'):
    # loop over all tweets posted by v:
    if direction == 'in':
        edges = v.in_edges()
    elif direction == 'out':
        edges = v.out_edges()
    else:
        raise Exception('Unknown direction')

    official_sources = [source_property_map[e] in \
                        official_twitter_clients_ids for e in edges]

    if len(official_sources) == 0:
        return np.nan
    else:
        return sum(official_sources) / len(official_sources)


def get_main_client(v, source_property_map, direction='in'):
    # returns the most used client use by user v (with direction='in')
    # or use to retweet v (with direction='out')

    # loop over all tweets posted by v:
    if direction == 'in':
        edges = v.in_edges()
    elif direction == 'out':
        edges = v.out_edges()
    else:
        raise Exception('Unknown direction')

    sources_names = Counter([source_global_map_id2sn[source_property_map[e]] for e in edges])

    if len(sources_names) == 0:
        return 'NA'
    else:
        return sources_names.most_common(1)[0][0]


def add_vertex_properties(G):
    # compute some vertex properties

    G.vp['k_out'] = G.degree_property_map('out')
    G.vp['k_in'] = G.degree_property_map('in')


# %%
# build and process each graph
t0 = time.time()
for media_type in edges_db_file[tweet_db_file1].keys():
    for graph_type in ['complete', 'simple']:
        for period in ['june-nov']:
            print(media_type)
            print(graph_type)
            print(period)

            edges_array = np.concatenate((edges_db_file[tweet_db_file1][media_type],
                                          edges_db_file[tweet_db_file2][media_type]))

            print(len(edges_array))
            G = build_graph_from_edges(edges_array, media_type)

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
            filename = os.path.join(save_dir, 'retweet_graph_' + media_type + '_' + \
                  graph_type + '_' + period + '.gt')
            print(filename)
            G.save(filename)
            print(time.time() - t0)

print('total time')
print(time.time() - t00)