import networkit as nk
import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
import numpy as np
from elites_2020_retweet_test import dask_load_tweets, apply_tweet_leanings
import pandas as pd
import pdb
import graph_tool as gt
import time
from graph_tool.draw import graphviz_draw

WORKING_DIR = '/home/crossb/working'
NUM_WORKERS = 32
THREADS_PER_WORKER = 2
Election_2020_dir = '/home/pub/hernan/Election_2020/joined_output'

BIAS_TO_RETWEET_NETWORKS = {
    'Center news': os.path.join(WORKING_DIR, '', 'Center news_retweet_edges.csv'),
    'Fake news': 'Fake news_retweet_edges.csv',
    'Extreme bias left': 'Extreme bias left_retweet_edges.csv',
    'Extreme bias right': 'Extreme bias right_retweet_edges.csv',
    'Left leaning news': 'Left leaning news_retweet_edges.csv',
    'Right leaning news': 'Right leaning news_retweet_edges.csv',
    'Left news': 'Left news_retweet_edges.csv',
    'Right news': 'Right news_retweet_edges.csv'
}


def initialize_local_cluster():
    """
    To parallelize our dask operations, intialize a local cluster and client.
    :return:
    """
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=THREADS_PER_WORKER,
                           scheduler_port=0, dashboard_address=None)
    client = Client(cluster)
    return client


def load_graphs_gt(path):
    fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    graphs = {}
    for file in fnames:
        bias = ' '.join(file.split('_')[:-2])
        print(bias)
        graph = gt.load_graph(os.path.join(path, file))
        graphs[bias] = graph
    return graphs


def load_from_graphtool(path):
    graph = nk.readGraph(path, nk.Format.GraphToolBinary)
    return graph


def top_influencers(N, graphs):
    """
    Gets the top N influencers of each graph in graphs dict
    :param num_influencers:
    :param graphs:
        A dictionary where key is the bias of the graph and the value is the graph-tool graph.
        *note: each vertex of the graph-tool graphs given has a CI property that was precalculated,
        giving us the vertex's rank.
    :return:
    """
    print("Get top influencers")
    stime = time.time()
    top_influencers_by_bias = {}
    for bias, graph in graphs.items():
        res = []
        for vertex in graph.vertices():
            res.append((graph.vp.CI_in[vertex], graph.vp.CI_out[vertex], graph.vp.user_id[vertex], vertex))
        res = sorted(res, key=lambda x: x[1], reverse=True)

        top_influencers_by_bias[bias] = [(x[3], x[2]) for x in res[:N]]
    print("Gathering top influencers took: {} seconds".format(time.time() - stime))
    return top_influencers_by_bias


def load_from_edgelist(path, directed=False):
    """
    Load our retweet network from their current format and convert the edgelist into a networkit graph.

    :param directed:
        Should the resulting network be directed? false results in undirected.
    :return:
    """

    with initialize_local_cluster() as client:
        #path = os.path.join(WORKING_DIR, 'url_classified_edgelists', 'Right news_edges.csv')
        edges = dd.read_csv(path, delimiter=',',
                            usecols=['auth_id', 'infl_id'],
                            dtype={'auth_id': np.int64, 'infl_id': np.int64}
                            )

        edges = edges.persist(retries=1000)
        progress(edges)
        edges = edges.compute()
        unique_uids = np.concatenate((edges['auth_id'].unique(), edges['infl_id'].unique()), axis=0)
        edges = edges.values

    uid_to_node = {uid: i for i, uid in enumerate(unique_uids)}
    node_to_uid = {i: uid for i, uid in enumerate(unique_uids)}

    graph = nk.graph.Graph(directed=directed)
    for edge in edges:
        (src_uid, dest_uid) = edge
        graph.addEdge(uid_to_node[src_uid], uid_to_node[dest_uid], addMissing=True)

    return graph, node_to_uid


def retweet_heirarchy(graph, hubs):
    """
    Find who is retweeting whom amongst our hubs. See if we can find any pattern
    :param graph:
    :param hubs:
    :return:
    """
    return


def top_influencer_network(bias_networks, influencers):
    """
    This method creates a single influencer network from the top influencer nodes / edges from
    each bias network.
    :param classified_edges:
    :param influencers:
    :return:
    """
    print("Create top influencer network")
    stime = time.time()
    leanings = sorted(list(influencers.keys()))

    influencer_edges = []
    user_to_bias = {uid: bias for bias, uids in influencers.items() for (vertex, uid) in uids}
    all_influencers = set(list(user_to_bias.keys()))
    for leaning in leanings:
        # grab all edges associated with the influencers of this category
        #influencer_uids = influencers[leaning]
        graph = bias_networks[leaning]
        for source, target, tweet_id in graph.iter_edges([graph.ep.tweet_id]):
            s_uid = graph.vp.user_id[source]
            t_uid = graph.vp.user_id[target]

            if s_uid in all_influencers and t_uid in all_influencers:
                influencer_edges.append((s_uid, t_uid, tweet_id, leaning))

    print("Num influencer edges:", len(influencer_edges))
    influencer_network = gt.Graph(directed=True)
    influencer_network.vertex_properties['user_id'] = influencer_network.new_vertex_property('int64_t')
    influencer_network.edge_properties['tweet_id'] = influencer_network.new_edge_property('int64_t')
    influencer_network.edge_properties['source_id'] = influencer_network.new_edge_property('int64_t')
    influencer_network.edge_properties['bias'] = influencer_network.new_edge_property('string')
    influencer_network.vp.user_id = influencer_network.add_edge_list(influencer_edges, hashed=True,
                                     eprops=[influencer_network.ep.tweet_id,
                                             influencer_network.ep.bias])#,
                                             #influencer_network.ep.source_id])

    # add a bias vertex property
    influencer_network.vertex_properties['bias'] = influencer_network.new_vertex_property('vector<int64_t>')
    influencer_network.vertex_properties['style'] = influencer_network.new_vertex_property('string')
    #vertex_to_pie_fractions = {}
    for v in influencer_network.vertices():
        #vertex_to_pie_fractions[v] = [0] * len(leanings)
        influencer_network.vp.style[v] = 'rounded'
        influencer_network.vp.bias[v] = [0] * len(leanings)
        for source, target, bias in influencer_network.iter_out_edges(v, [influencer_network.ep.bias]):
            #bias = influencer_network.ep.bias[edge]
            leaning_idx = leanings.index(bias)
            uid = influencer_network.vp.user_id[v]
            influencer_network.vp.bias[v][leaning_idx] += 1

    print("Creating top influencer network took: {} seconds".format(time.time() - stime))
    #return influencer_edges
    return influencer_network


def hub_analysis(graph, node_to_uid):
    """
    Find the hubs in our given network and their neighborhood.
    :param graph:
        networkit graph
    :param node_to_uid:
        networkit graph node number to twitter user id
    :return:
    """
    # TODO: Networkit might have a method to find neighbors within some distance.
    # get hubs
    NUM_HUBS = 100
    node_degrees = np.array([[node, graph.degreeOut(node)] for node in node_to_uid.keys()])

    # top 20
    hubs = node_degrees[node_degrees[:,1].argsort()][::-1][:NUM_HUBS]
    hubs_uid = [(node_to_uid[node], degree) for (node, degree) in hubs]
    print("Hubs: {}".format(hubs))

    #hub_nbrs = {node: [x for x in graph.iterNeighbors(node)] for node, _ in hubs}
    #print("Hub neighbors")
    #for node, nbrs in hub_nbrs.items():
    #    print("{} neighbors: {}".format(node, nbrs))

    return hubs_uid


def time_series(df, rule='1D'):
    """
    We expect the dataframe to have a column named 'timestamp' that we will use in time-series analysis.
    :param df:
    :return:
    """
    # df.resample(rule, on='timestamp').index.nunique()
    return df.resample(rule).nunique()['id'].compute()


def political_leaning_time_series_analysis():
    """
    Recreate figure 5b from the 2016 paper. Perform time-series analysis on the leanings and stance data
    then correlate them and draw the figure.
    :return:
    """
    with initialize_local_cluster() as client:
        # get tweets assigned by leanings (plus user_id)
        tweets = remove_retweets(dask_load_tweets(Election_2020_dir)).set_index('id')
        tweets = apply_tweet_leanings(tweets)

        # get any desired statistics about the twitter data (num unique users / tweets)

        # time series binning based on leaning
        # create a column per leaning
        tweet_ts_no_retweet = pd.DataFrame()
        for leaning in tweets.keys():
            tweet_ts_no_retweet[leaning] = time_series(tweets[leaning])

        # at this point, we have columns corresponding to the instances of tweets by each bias in daily chunks


        pdb.set_trace()
        # get tweets by stance
        for stance in ['biden', 'trump']:
            tweet_ts_no_retweet[leaning] = time_series(tweets[stance])

        # pyloess STL

        # Draw
    return


def remove_retweets(tweets):
    """
    From our entire tweet set, remove any tweets that are flagged as retweets.
    :param tweets:
        dask dataframe with the following columns: id, user_id, and timestamp.
            note: id = tweet_id, timestamp = time tweet was created.
    :return:
    """
    retweet_dir = os.path.join(Election_2020_dir, 'retweets')
    #columns = ['id', 'user_id', 'retweet_id']
    retweet_ids = dd.read_csv(os.path.join(retweet_dir, '*.csv'), delimiter=',', usecols=['retweet_id'],
                       dtype={'retweet_id': str}).rename(columns={'retweet_id': 'id'})
    convert_to_int = lambda df: pd.to_numeric(df, errors='coerce')
    retweet_ids['id'] = retweet_ids['id'].map_partitions(convert_to_int)
    retweet_ids = retweet_ids.dropna()['id'].unique().astype(np.int64).persist(retries=100)
    progress(retweet_ids)
    print("Loaded Retweets")

    #tweets_no_retweet = tweets.merge(retweet_ids, left_index=True, right_index=True, indicator=True, how='outer').persist(retries=100)
    #progress(tweets_no_retweet)
    #print("Merged")
    #tweets_no_retweet = tweets_no_retweet[tweets_no_retweet['_merge'] == 'left_only'].drop(['_merge'], axis=1)
    #retweet_ids = retweet_ids['id'].compute().tolist()
    tweets = tweets.reset_index()
    tweets_no_retweet = tweets[~tweets['id'].isin(retweet_ids.compute())].persist(retries=100)
    #tweets_no_retweet = tweets_no_retweet.persist(retries=100)
    progress(tweets_no_retweet)
    print("Removed all retweets from tweet collection.")

    return tweets_no_retweet


def influencer_network_anlaysis():
    #network_base_path = '/home/crossb/working/url_classified_edgelists'
    #network_biases = ['Extreme bias left_edges.csv', 'Left news_edges.csv', 'Left leaning news_edges.csv',
    #                  'Center news_edges.csv', 'Fake news_edges.csv',
    #                  'Extreme bias right_edges.csv', 'Right news_edges.csv', 'Right leaning news_edges.csv']
    #biased_edgelists = {}
    #for bias in network_biases:
    #    filename = "{}_edges.csv".format(bias)
    #    path = os.path.join(network_base_path, filename)
    #    biased_edgelists[bias] = read_edgeslist(path)
    path_2020 = '/home/crossb/packaged_ci/graphs/2020/'
    top_n_influencers = 100
    biased_graphs = load_graphs_gt(path_2020)
    biased_influencers = top_influencers(top_n_influencers, biased_graphs)
    influencer_network = top_influencer_network(biased_graphs, biased_influencers)
    #influencer_network.save(os.path.join(WORKING_DIR, 'influencer_network.gt'))
    stats = network_characteristics_gt(influencer_network)
    print("Influencer network stats")
    for stat, value in stats.items():
        print("{}: {}".format(stat, value))

    #most_infl_influencers = top_influencers(10, {'top': influencer_network})
    #print("Most influential:", most_infl_influencers)

    # draw influencer network, color nodes by the bias property
    # for each vertex, get the counts of edge class membership.
    graphviz_draw(influencer_network, ratio="expand",
                  vprops={'style': influencer_network.vp.style},
                  #vprops={'style': "wedged"},
                  gprops={'scale': 20},
                  output='top_influencers.svg')


    return


def network_characteristics_nk(graph):
    print("Gathering network statistics!")
    characteristics = {
        'n_nodes': graph.numberOfNodes(),
        'n_edges': graph.numberOfEdges(),
        'avg_degree': 0,
        'max_out_degree': 0,
        'max_in_degree': 0,
        'in_heterogeneity': 0,
        'out_heterogeneity': 0
    }

    # get in-degree distribution
    in_degrees = np.array([graph.degreeIn(node) for node in graph.iterNodes()])
    # get out-degree distribution
    out_degrees = np.array([graph.degreeOut(node) for node in graph.iterNodes()])

    characteristics['avg_degree'] = np.average([graph.degree(node) for node in graph.iterNodes()])
    characteristics['max_in_degree'] = np.max(in_degrees)
    characteristics['max_out_degree'] = np.max(out_degrees)

    characteristics['in_heterogeneity'] = np.std(in_degrees) / characteristics['avg_degree']
    characteristics['out_heterogeneity'] = np.std(out_degrees) / characteristics['avg_degree']
    return characteristics


def network_characteristics_gt(graph):
    print("Gathering network statistics!")
    characteristics = {
        'n_nodes': graph.num_vertices(),
        'n_edges': graph.num_edges(),
        'avg_degree': 0,
        'max_out_degree': 0,
        'max_in_degree': 0,
        'in_heterogeneity': 0,
        'out_heterogeneity': 0
    }
    nodes = graph.get_vertices()
    # get in-degree distribution
    in_degrees = graph.get_in_degrees(nodes)
    # get out-degree distribution
    out_degrees = graph.get_out_degrees(nodes)

    characteristics['avg_degree'] = np.average(graph.get_total_degrees(nodes))
    characteristics['max_in_degree'] = np.max(in_degrees)
    characteristics['max_out_degree'] = np.max(out_degrees)


    # TODO: NEED TO CHANGE TO REFLECT HOW IT WAS DONE IN PREVIOUS PAPER
    characteristics['in_heterogeneity'] = np.std(in_degrees) / characteristics['avg_degree']
    characteristics['out_heterogeneity'] = np.std(out_degrees) / characteristics['avg_degree']
    return characteristics


def all_network_stats():
    graphs = load_graphs_gt('/home/crossb/packaged_ci/graphs/2020/')

    network_stats = {}
    for bias, graph in graphs.items():
        network_stats[bias] = network_characteristics_gt(graph)

    for bias, characteristics in network_stats.items():
        print("{} Network Statistics".format(bias))
        for stat, value in characteristics.items():
            print("{}: {}".format(stat, value))
        print("===========================================")
    return


def read_edgeslist(path, columns=('auth_id', 'infl_id')):
    edges = dd.read_csv(path, delimiter=',',
                        usecols=columns,
                        dtype={col: np.int64 for col in columns}
                        )
    return edges


def main():
    #graph = load_from_graphtool('/home/crossb/packaged_ci/graphs/2020/Right_news_2020_ci.gt')
    #graphs = load_graphs_gt('/home/crossb/packaged_ci/graphs/2020/')
    influencer_network_anlaysis()

    #for bias, path in BIAS_TO_RETWEET_NETWORKS.items():
    #    graph, node_to_uid = load_from_edgelist(path, directed=True)
    #    #hub_nodes = hub_analysis(graph, node_to_uid)
    #    network_stats[bias] = network_characteristics(graph)
    #    # create top influencer network
#
    #    #political_leaning_time_series_analysis()



    return


if __name__ == '__main__':
    main()