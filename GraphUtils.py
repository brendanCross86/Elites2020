# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

#import csv
from multiprocessing import Pool
#import multiprocessing as mp
import pandas as pd
import os
from networkx import MultiDiGraph, write_graphml
try:
    import graph_tool.all as gt
except:
    pass
import numpy as np
import time
#import pdb
#from math import ceil
#import concurrent.futures
#from datetime import datetime



#os.system("taskset -p 0xff %d" % os.getpid())

    
def buildGraphSqlite(conn, graph_type, start_date, stop_date, 
                       hashtag_list_filter=None,
                       keyword_list_filter=None,
                       save_filename=None,
                       ht_group_supporters_include=None,
                       ht_group_supporters_exclude=None,
                       queries_selected=None,
                       additional_sql_select_statement=None,
                       graph_lib='graph_tool'):
    """ Returns graph for interaction types in `graph_type` from sqldatabase,
        using the graph library graph_lib.
        
        Notes
        -----
        tweets are selected such that `start_date` <= tweet timestamp < `stop_date`.
        
        if hashtag_list_filter is provided, only tweets containing hashtags in
        this list are used.

        if keyword_list_filter is provided, only tweets containing keywords in
        this list are used.
        
        `ht_group_supporters_include` and `ht_group_supporters_exclude` are used
        to select tweets with hashtags belonging to certain groups
        
        `additional_sql_select_statement` can be use to add a condition of the 
        tweet ids. Must start by "SELECT tweet_id FROM ... "
        
        user_ids are stored in a internal vertex_property_map named `user_id`.
        
        tweet_ids are stored in a internal edge_property_map names `tweet_id`.
        
        `graph_lib` can be 'graph_tool', 'networkx' or 'edge_list', where
        edge_list returns a numpy array of edges with (influencer_id, tweet_author_id, tweet_id)
        
    """
    
    c = conn.cursor()

    
    # transform the list of graph types to a list of table names
    graph_type_table_map = {'retweet': 'tweet_to_retweeted_uid',
                            'reply' : 'tweet_to_replied_uid',
                            'mention' : 'tweet_to_mentioned_uid',
                            'quote' : 'tweet_to_quoted_uid'}

    # table_name to influencer col_name                            
    table_to_col_map = {'tweet_to_retweeted_uid' : 'retweeted_uid',
                            'tweet_to_replied_uid': 'replied_uid',
                            'tweet_to_mentioned_uid' : 'mentioned_uid',
                            'tweet_to_quoted_uid' : 'quoted_uid'}  
                  
    table_names = []                            
    if isinstance(graph_type, str):
        if graph_type == 'all':
            table_names = list(graph_type_table_map.values())
        else:
            graph_type = [graph_type]
        
    if isinstance(graph_type, list):
        for g_type in graph_type:
            if g_type in graph_type_table_map.keys():
                table_names.append(graph_type_table_map[g_type])
            else:
                raise ValueError('Not implemented graph_type')
    
        
        
    table_queries = []
    values = []
    for table in table_names:
        
        sql_select = """SELECT tweet_id, {col_name}, author_uid
                     FROM {table} 
                     WHERE tweet_id IN 
                         (
                         SELECT tweet_id 
                         FROM tweet 
                         WHERE datetime_EST >= ? AND datetime_EST < ?
                         )""".format(table=table, col_name=table_to_col_map[table])
                         
        
        values.extend([start_date, stop_date])
    
        # add conditions on hashtags
        if hashtag_list_filter is not None:
            
            sql_select = '\n'.join([sql_select,
                    """AND tweet_id IN
                     (
                     SELECT tweet_id
                     FROM hashtag_tweet_user
                     WHERE hashtag IN ({seq})
                     )
                     """\
                     .format(seq = ','.join(['?']*len(hashtag_list_filter)))])
            
            for ht in hashtag_list_filter:
                values.append(ht)
    
        # add conditon on keyword
        if keyword_list_filter is not None:                             
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         SELECT tweet_id
                         FROM tweet_to_keyword
                         WHERE keyword IN ({seq})
                         )
                         """\
                         .format(seq = ','.join(['?']*len(keyword_list_filter)))])                      
                             
            for kw in keyword_list_filter:
                values.append(kw)
                
        if additional_sql_select_statement is not None:
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         """ + additional_sql_select_statement + """
                         )
                         """])
        #                 
        # intersect with given ht groups
        #
        if ht_group_supporters_include is not None:
            sql_included_groups = []
            for ht_gn in ht_group_supporters_include:
                sql_included_groups.append("""SELECT tweet_id 
                                        FROM hashtag_tweet_user
                                        WHERE ht_group == '{htgn}'""".format(htgn=ht_gn))
                
            sql_included_groups = '\nUNION ALL\n'.join(sql_included_groups)
                
            
            if ht_group_supporters_exclude is not None:
                sql_excluded_groups = []
                for ht_gn in ht_group_supporters_exclude:
                    sql_excluded_groups.append("""SELECT tweet_id 
                                    FROM hashtag_tweet_user
                                    WHERE ht_group == '{htgn}'""".format(htgn=ht_gn))
        
                sql_excluded_groups = '\nUNION ALL\n'.join(sql_excluded_groups)

            
                sql_select = '\n'.join([sql_select,
                            """AND tweet_id IN
                             (
                             SELECT * FROM (""" + sql_included_groups + """)
                             EXCEPT
                             SELECT * FROM (""" + sql_excluded_groups + """)
                             )
                             """])
                             
            else:
                sql_select = '\n'.join([sql_select,
                            """AND tweet_id IN
                             (
                             """ + sql_included_groups + """
                             )
                             """])
                         
        #
        # intersect with queries selected
        #
        if queries_selected is not None:                          
            sql_select = '\n'.join([sql_select,
                        """AND tweet_id IN
                         (
                         SELECT tweet_id
                         FROM tweet_to_query_id
                         WHERE query_id IN (
                                            SELECT id 
                                            FROM query 
                                            WHERE query IN ('{qulst}')
                                            )
                         )
                         """\
                         .format(qulst = "','".join(queries_selected))])                      
                             

            
        table_queries.append(sql_select)
        
    # take union of all the interaction type tables
    sql_query = '\nUNION \n'.join(table_queries)
                              
#    print(sql_query)                
    c.execute(sql_query, values)
                     
    if graph_lib == 'graph_tool':
        G = gt.Graph(directed=True)
        G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
        G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')
    
        edge_list = np.array([(infl_uid, auth_uid, tweet_id ) for tweet_id, 
                              infl_uid, auth_uid in c.fetchall()],
                              dtype=np.int64)
    
        G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id])
    
            
        if save_filename is not None:
            G.save(save_filename)
            
    elif graph_lib == 'networkx':
        G = MultiDiGraph(graph_type=', '.join(graph_type))
        
        
        G.add_edges_from([(infl_uid, auth_uid, {'tweet_id': tweet_id}) for tweet_id, 
                              infl_uid, auth_uid in c.fetchall()])
        
        if save_filename is not None:
            write_graphml(G, save_filename)
            
    elif graph_lib == 'edge_list':
        G = np.array([(infl_uid, auth_uid, tweet_id ) for tweet_id, 
                              infl_uid, auth_uid in c.fetchall()],
                              dtype=np.int64)
        
        
    return G


def buildGraphSqlite2020(conn, graph_type, start_date, stop_date,
                     hashtag_list_filter=None,
                     keyword_list_filter=None,
                     save_filename=None,
                     ht_group_supporters_include=None,
                     ht_group_supporters_exclude=None,
                     queries_selected=None,
                     additional_sql_select_statement=None,
                     graph_lib='graph_tool'):
    """ Returns graph for interaction types in `graph_type` from sqldatabase,
        using the graph library graph_lib.

        Notes
        -----
        tweets are selected such that `start_date` <= tweet timestamp < `stop_date`.

        if hashtag_list_filter is provided, only tweets containing hashtags in
        this list are used.

        if keyword_list_filter is provided, only tweets containing keywords in
        this list are used.

        `ht_group_supporters_include` and `ht_group_supporters_exclude` are used
        to select tweets with hashtags belonging to certain groups

        `additional_sql_select_statement` can be use to add a condition of the
        tweet ids. Must start by "SELECT tweet_id FROM ... "

        user_ids are stored in a internal vertex_property_map named `user_id`.

        tweet_ids are stored in a internal edge_property_map names `tweet_id`.

        `graph_lib` can be 'graph_tool', 'networkx' or 'edge_list', where
        edge_list returns a numpy array of edges with (influencer_id, tweet_author_id, tweet_id)

    """

    c = conn.cursor()

    # transform the list of graph types to a list of table names
    graph_type_table_map = {'retweet': 'tweet_to_retweeted_uid',
                            'reply': 'tweet_to_replied_uid',
                            'mention': 'tweet_to_mentioned_uid',
                            'quote': 'tweet_to_quoted_uid'}

    # table_name to influencer col_name
    table_to_col_map = {'tweet_to_retweeted_uid': 'retweeted_uid',
                        'tweet_to_replied_uid': 'replied_uid',
                        'tweet_to_mentioned_uid': 'mentioned_uid',
                        'tweet_to_quoted_uid': 'quoted_uid'}

    table_names = []
    if isinstance(graph_type, str):
        if graph_type == 'all':
            table_names = list(graph_type_table_map.values())
        else:
            graph_type = [graph_type]

    if isinstance(graph_type, list):
        for g_type in graph_type:
            if g_type in graph_type_table_map.keys():
                table_names.append(graph_type_table_map[g_type])
            else:
                raise ValueError('Not implemented graph_type')

    table_queries = []
    values = []
    for table in table_names:
        sql_select = """
        SELECT tweet.tweet_id, tweet.user_id, {table}.author_uid
        FROM tweet
        JOIN {table} ON {table}.tweet_id = tweet.tweet_id
        WHERE tweet.tweet_id IN (
            SELECT tweet_id
            FROM tweet
            WHERE datetime_EST >= {start_time} AND datetime_EST < {stop_time}
        )
        """.format(table=table, start_time=start_date, stop_time=stop_date)

        # add conditions on hashtags
        if hashtag_list_filter is not None:

            sql_select = '\n'.join([sql_select,
                                    """AND tweet_id IN
                                     (
                                     SELECT tweet_id
                                     FROM hashtag_tweet_user
                                     WHERE hashtag IN ({seq})
                                     )
                                     """ \
                                   .format(seq=','.join(['?'] * len(hashtag_list_filter)))])

            for ht in hashtag_list_filter:
                values.append(ht)


        if additional_sql_select_statement is not None:
            sql_select = '\n'.join([sql_select,
                                    """AND tweet_id IN
                                     (
                                     """ + additional_sql_select_statement + """
                         )
                         """])

        #
        # intersect with queries selected
        #
        if queries_selected is not None:
            sql_select = '\n'.join([sql_select,
                                    """AND tweet_id IN
                                     (
                                     SELECT tweet_id
                                     FROM tweet_to_query_id
                                     WHERE query_id IN (
                                                        SELECT id 
                                                        FROM query 
                                                        WHERE query IN ('{qulst}')
                                                        )
                                     )
                                     """ \
                                   .format(qulst="','".join(queries_selected))])

        table_queries.append(sql_select)

    # take union of all the interaction type tables
    sql_query = '\nUNION \n'.join(table_queries)

    #    print(sql_query)
    c.execute(sql_query, values)

    if graph_lib == 'graph_tool':
        G = gt.Graph(directed=True)
        G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
        G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')

        edge_list = np.array([(infl_uid, auth_uid, tweet_id) for tweet_id,
                                                                 infl_uid, auth_uid in c.fetchall()],
                             dtype=np.int64)

        G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id])

        if save_filename is not None:
            G.save(save_filename)

    elif graph_lib == 'networkx':
        G = MultiDiGraph(graph_type=', '.join(graph_type))

        G.add_edges_from([(infl_uid, auth_uid, {'tweet_id': tweet_id}) for tweet_id,
                                                                           infl_uid, auth_uid in c.fetchall()])

        if save_filename is not None:
            write_graphml(G, save_filename)

    elif graph_lib == 'edge_list':
        G = np.array([(infl_uid, auth_uid, tweet_id) for tweet_id,
                                                         infl_uid, auth_uid in c.fetchall()],
                     dtype=np.int64)

    return G




def loadRetweetGraphData(data_path, start_date, stop_date, threads=4):
    """

    :param data_path:
    :param start_date:
    :param stop_date:
    :param threads:
    :return:
    """
    data_dirs = [
        os.path.join(data_path, 'retweets'),
        # os.path.join(data_path, 'replies'),
        os.path.join(data_path, 'user_mentions'),
        os.path.join(data_path, 'quotes'),
    ]
    data_dir_to_cols = {
        os.path.join(data_path, 'retweets'): ['id', 'user_id'],
        os.path.join(data_path, 'user_mentions'): ['id', 'user_id'],
        os.path.join(data_path, 'quotes'): ['id', 'user_id'],
    }
    cols_to_dtype = {
        'id': np.int64,
        'user_id': np.int64
    }

    tweet_dir = os.path.join(data_path, 'tweets')

    start_time = time.time()
    # gather all the tweet data that exists in our start / stop range
    merged_data = []
    tweet_data = pd.concat([
        pd.read_csv(os.path.join(tweet_dir, fp), delimiter=',', parse_dates=['created_at'],
                    dtype={'id': str, 'created_at': str, 'user_id': str})
        for fp in os.listdir(tweet_dir)
    ])
    tweet_data['id'] = pd.to_numeric(tweet_data['id'], errors='coerce')
    tweet_data['user_id'] = pd.to_numeric(tweet_data['user_id'], errors='coerce')
    tweet_data = tweet_data[(tweet_data['id'].notna() == True) & (tweet_data['user_id'].notna() == True)]

    # filter tweet_data on the datetime column
    tweet_data = tweet_data[(tweet_data['created_at'] >= start_date)
                            & (tweet_data['created_at'] < stop_date)]
    print("Tweet read time: {} seconds".format(time.time() - start_time))

    # if we want to filter on political stance, we run our political stance query and filter out tweet data
    # that doesn't match.
    # TODO: Improve how we query, should be able to do something like specify a list of query methods to
    #  apply and then query statements to use to merge / filter
    #if stance is not None:
    #    left_data, right_data = political_stance_query(data_path, threads)
    #    if stance == 'left':
    #        tweet_data = tweet_data[tweet_data['id'].isin(left_data['id'])]
    #    else:
    #        tweet_data = tweet_data[tweet_data['id'].isin(right_data['id'])]

    for dir in data_dirs:
        print("Starting {}".format(dir))
        start_time = time.time()
        data = []
        for fp in os.listdir(dir):
            with open(os.path.join(dir, fp), newline='') as file:
                data.append(pd.read_csv(file, delimiter=',', usecols=data_dir_to_cols[dir]))
        data = pd.concat(data)
        data.rename(columns={'user_id': 'infl_uid'}, inplace=True)

        merged_data.append(pd.merge(tweet_data, data, on='id'))
        print("{} read time taken: {} seconds".format(dir, time.time() - start_time))
    full_data = pd.concat(merged_data)
    return full_data


def dataToGraphCSV(data, graph_type, save_filename=None, graph_lib='graph_tool'):
    """
    Take our pandas dataframe containing all of our edge data and create the desired graph output.
    :param data:
        pandas data containing three columns, tweet id, author id, influenced user id
    :param graph_type:
    :param save_filename:
    :param graph_lib:
    :return:
    """
    # now that we have our data, we can output
    if graph_lib == 'graph_tool':
        G = gt.Graph(directed=True)
        G.vertex_properties['user_id'] = G.new_vertex_property('int64_t')
        G.edge_properties['tweet_id'] = G.new_edge_property('int64_t')

        edge_list = np.array([(infl_uid, auth_uid, tweet_id) for tweet_id,
                                                                 infl_uid, auth_uid in data.iterrows()],
                             dtype=np.int64)

        G.vp.user_id = G.add_edge_list(edge_list, hashed=True, eprops=[G.ep.tweet_id])

        if save_filename is not None:
            G.save(save_filename)

    elif graph_lib == 'networkx':
        G = MultiDiGraph(graph_type=', '.join(graph_type))

        G.add_edges_from([(infl_uid, auth_uid, {'tweet_id': tweet_id}) for tweet_id,
                                                                           infl_uid, auth_uid in data.iterrows()])

        if save_filename is not None:
            write_graphml(G, save_filename)

    elif graph_lib == 'edge_list':
        G = np.array([(infl_uid, auth_uid, tweet_id) for idx, (tweet_id, created, auth_uid,
                                                               infl_uid) in data.iterrows()],
                     dtype=np.int64)
    return G


def buildGraphCSV(data_path, graph_type, start_date, stop_date,
                  additional_query=None,
                  stance=None,
                  save_filename=None,
                  graph_lib='graph_tool',
                  threads=4):
    """

    :param data_path:
        Directory containing all of our csv data folders.
    :param graph_type:
    :param start_date:
    :param stop_date:
    :param additional_query:
    :param stance:
        Temporary argument to flag for filtering on political stance. This will be replaced with a better query
        method when time allows
    :param save_filename:
    :param graph_lib:
    :param threads:
    :return:
    """
    #pdb.set_trace()
    # For simplicity we will expect our data directory to be structured in a particular fashion
    full_data = loadRetweetGraphData(data_path, start_date, stop_date, threads)

    # now that we have our data, we can output
    G = dataToGraphCSV(full_data, graph_type, save_filename, graph_lib)

    return G
