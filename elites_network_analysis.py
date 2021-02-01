import networkit as nk
import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
import numpy as np
from elites_2020_retweet_test import dask_load_tweets, apply_tweet_leanings
import pandas as pd
import pdb
import graph_tool.all as gt
import time
import operator
from matplotlib import cm
import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.stats import sem
#from graph_tool.draw import graphviz_draw
#from graph_tool.all import

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
BIAS_TO_COLOR = {
    'center': 'mediumseagreen',
    'fake': 'saddlebrown',
    'left extreme': 'tab:pink',#'deeppink',#'hotpink',
    'left leaning': 'mediumblue',#'royalblue',
    'left': 'darkmagenta',#'purple',
    'right leaning': 'darkgreen',
    'right': 'orange',
    'right extreme': 'tab:red'
}

RT_GRAPHS_DIR_2016 = '/home/crossb/packaged_ci/graphs/2016/'
PATH_TO_BIAS_2016 = {
    os.path.join(RT_GRAPHS_DIR_2016, 'center_2016_ci.gt'): 'center',
    os.path.join(RT_GRAPHS_DIR_2016, 'extreme_bias_left_2016_ci.gt'): 'left extreme',
    os.path.join(RT_GRAPHS_DIR_2016, 'extreme_bias_right_2016_ci.gt'): 'right extreme',
    os.path.join(RT_GRAPHS_DIR_2016, 'fake_2016_ci.gt'): 'fake',
    os.path.join(RT_GRAPHS_DIR_2016, 'lean_left_2016_ci.gt'): 'left leaning',
    os.path.join(RT_GRAPHS_DIR_2016, 'lean_right_2016_ci.gt'): 'right leaning',
    os.path.join(RT_GRAPHS_DIR_2016, 'left_2016_ci.gt'): 'left',
    os.path.join(RT_GRAPHS_DIR_2016, 'right_2016_ci.gt'): 'right'
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


def load_graphs_gt(path, year='2020'):
    fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    graphs = {}
    for file in fnames:

        if 'pro_' in file or 'combined' in file or 'complete' in file:
            continue
        if year == '2020':
            #bias = ' '.join(file.split('_')[:-2])
            bias = ' '.join(file.split('_')[:-3]) # temporarily, since we have both complete and simple in one dir,
            # we will name the bias the full path so that we can get them all in one table
        elif year == '2016':
            bias = PATH_TO_BIAS_2016[os.path.join(path, file)]
        else:
            raise ValueError

        print(bias)
        graph = gt.load_graph(os.path.join(path, file))
        graphs[bias] = graph
    return graphs



def merge_graphs(bias_networks):
    return



def load_graphs_nk(path):
    fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    graphs = {}
    for file in fnames:
        if 'pro_' in file:
            continue
        bias = ' '.join(file.split('_')[:-2])
        print(bias)
        graph = load_from_graphtool_nk(os.path.join(path, file))
        graphs[bias] = graph
    return graphs


def load_from_graphtool_nk(path):
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

        top_influencers_by_bias[bias] = {rank+1: (x[3], x[2], x[1]) for rank, x in enumerate(res[:N])}
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


def top_influencer_network(bias_networks, influencers, keep_all_edges=False):
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
    user_to_bias = {uid: bias for bias, uids in influencers.items() for rank, (vertex, uid, ci) in uids.items()}

    all_influencers = set(list(user_to_bias.keys()))
    total_edges = 0
    accepted_edges = 0
    for leaning in leanings:
        # grab all edges associated with the influencers of this category
        #influencer_uids = influencers[leaning]
        graph = bias_networks[leaning]
        for source, target, tweet_id in graph.iter_edges([graph.ep.tweet_id]):
            s_uid = graph.vp.user_id[source]
            t_uid = graph.vp.user_id[target]

            if keep_all_edges or (s_uid in all_influencers and t_uid in all_influencers):
                influencer_edges.append((s_uid, t_uid, tweet_id, leaning, BIAS_TO_COLOR[leaning]))
                accepted_edges += 1
            total_edges += 1
    print("Total Edges considered:", total_edges)
    print("Edges accepted:", accepted_edges)

    print("Num influencer edges:", len(influencer_edges))
    influencer_network = gt.Graph(directed=True)
    influencer_network.vertex_properties['user_id'] = influencer_network.new_vertex_property('int64_t')
    influencer_network.edge_properties['tweet_id'] = influencer_network.new_edge_property('int64_t')
    influencer_network.edge_properties['source_id'] = influencer_network.new_edge_property('int64_t')
    influencer_network.edge_properties['bias'] = influencer_network.new_edge_property('string')
    influencer_network.edge_properties['color'] = influencer_network.new_edge_property('string')
    influencer_network.vp.user_id = influencer_network.add_edge_list(influencer_edges, hashed=True,
                                     eprops=[influencer_network.ep.tweet_id,
                                             influencer_network.ep.bias,
                                             influencer_network.ep.color])#,
                                             #influencer_network.ep.source_id])

    uid_to_new_vertex = {influencer_network.vp.user_id[v]: v for v in influencer_network.vertices()}

    # add a bias vertex property
    influencer_network.vertex_properties['pie_fractions'] = influencer_network.new_vertex_property('vector<int64_t>')
    influencer_network.vertex_properties['shape'] = influencer_network.new_vertex_property('string')
    influencer_network.vertex_properties['text'] = influencer_network.new_vertex_property('string')
    influencer_network.vertex_properties['text_color'] = influencer_network.new_vertex_property('string')
    influencer_network.vertex_properties['size'] = influencer_network.new_vertex_property('double')
    influencer_network.vertex_properties['pie_colors'] = influencer_network.new_vertex_property('vector<string>')
    # calculate the highest rank each vertex got
    #vertex_ranks = {v: {bias: 0 for bias in leanings} for v in influencer_network.vertices()}
    vertex_ranks = {}
    vertex_to_bias_ci = {}
    for bias, uids in influencers.items():
        for rank, (old_v, uid, ci) in uids.items():
            if uid not in uid_to_new_vertex.keys():
                continue
            vertex = uid_to_new_vertex[uid]

            if vertex not in vertex_to_bias_ci or vertex not in vertex_ranks:
                #vertex_ranks[vertex] = [(rank, bias)]
                #vertex_ranks[vertex] = {bias: rank}
                vertex_ranks[vertex] = {bias: 0 for bias in leanings}
                vertex_to_bias_ci[vertex] = {bias: ci}
            else:
                #vertex_ranks[vertex].append((rank, bias))
                vertex_to_bias_ci[vertex][bias] = ci

            vertex_ranks[vertex][bias] = rank

    largest_ci_by_bias = {bias: max([ci for rank, (vertex, uid, ci) in uids.items()]) for bias, uids in influencers.items()}

    for v in influencer_network.vertices():
        # highest rank is a tuple (rank, bias)
        #(highest_rank, bias) = sorted(vertex_ranks[v].values(), reverse=True)[0]
        try:
            (bias, highest_rank) = max([x for x in vertex_ranks[v].items() if x[1] != 0], key=operator.itemgetter(1))
        except KeyError:
            pdb.set_trace()
        if highest_rank <= 5:
            influencer_network.vp.text[v] = '{}'.format(highest_rank)
        else:
            influencer_network.vp.text[v] = ''
        influencer_network.vp.text_color[v] = BIAS_TO_COLOR[bias]
        influencer_network.vp.shape[v] = 'pie'

        influencer_network.vp.pie_colors[v] = [BIAS_TO_COLOR[bias] for bias in leanings]
        influencer_network.vp.pie_fractions[v] = [vertex_ranks[v][bias] for bias in leanings]

        influencer_network.vp.size[v] = 7 + (vertex_to_bias_ci[v][bias] / largest_ci_by_bias[bias] * 13)

        #influencer_network.vp.pie_fractions[v] = [0] * len(leanings)
        #for source, target, bias in influencer_network.iter_out_edges(v, [influencer_network.ep.bias]):
        #    leaning_idx = leanings.index(bias)
        #    uid = influencer_network.vp.user_id[v]
        #    influencer_network.vp.pie_fractions[v][leaning_idx] += 1

        # convert the bias property into fractions of the total
        total = sum(influencer_network.vp.pie_fractions[v])
        #ranks = np.array([1 - (x/total) if x != 0 else 0 for x in influencer_network.vp.pie_fractions[v]])
        ranks = np.array([1/x if x != 0 else 0 for x in influencer_network.vp.pie_fractions[v]])
        fractions = ranks/np.sum(ranks) * 100
        #if np.sum(fractions) > 1:
        #    fractions[np.where(fractions != 0)[0][0]] -= (np.sum(fractions) - 1)
        #    assert(np.sum(fractions) == 1)
        influencer_network.vp.pie_fractions[v] = list(fractions.astype(np.int64))

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


def influencer_network_anlaysis(year=2020):
    path_2016 = '/home/crossb/packaged_ci/graphs/2016/'
    path_2020 = '/home/crossb/packaged_ci/graphs/2020/'
    top_n_influencers = 30
    biased_graphs = load_graphs_gt(path_2020)
    biased_influencers = top_influencers(top_n_influencers, biased_graphs)
    influencer_network = top_influencer_network(biased_graphs, biased_influencers)
    #influencer_network.save(os.path.join(WORKING_DIR, 'influencer_network.gt'))
    stats = network_characteristics_gt(influencer_network)
    print("Influencer network stats")
    for stat, value in stats.items():
        print("{}: {}".format(stat, value))

    most_infl_influencers = top_influencers(10, {'top': influencer_network})
    print("Most influential:", most_infl_influencers)


    # save influencer network
    gt.save('data/2020/influencer_network.gt')

    # networkit stats
    nk_graph = load_from_graphtool_nk('data/2020/influencer_network.gt')
    characteristics = network_characteristics_nk(nk_graph, 10)

    for stat, value in characteristics.items():
        if "centrality" in stat:
            print("{}: {}".format(
                stat,
                ','.join(['(Node: {}: {})'.format(influencer_network.vp.user_id[n], v) for (n, v) in value])))
        else:
            print("{}: {}".format(stat, value))

    # Draw with the vertices as pie charts
    vprops = {'pie_fractions': influencer_network.vp.pie_fractions,
              'shape': influencer_network.vp.shape,
              'text': influencer_network.vp.text,
              'text_color': influencer_network.vp.text_color,
              'size': influencer_network.vp.size,
              'pie_colors': influencer_network.vp.pie_colors,
              'text_position': 200,
              'font_size': 14,
              'text_offset': [0.0, 1.0]
              }
    eprops = {'color': 'lightgray'}
    # r=1000 leads to cool graph
    #pos = gt.fruchterman_reingold_layout(influencer_network, r=35.0, circular=True, n_iter=2000)
    pos = gt.arf_layout(influencer_network, d=4, max_iter=0)
    gt.graph_draw(influencer_network, pos=pos, vprops=vprops,
                  eprops=eprops, output='top_influencers.svg')
    #with cairo.SVGSurface('top_influencers.svg', 1024, 1280) as surface:
    #    cr = cairo.Context(surface)
    #    gt.cairo_draw(influencer_network, pos, cr, vprops=vprops, eprops=eprops,
    #                  ecmap=(cm.get_cmap(), 0.2),
    #                  output='top_influencers.svg')

    return


def network_characteristics_nk(graph, N):
    print("Gathering network statistics!")
    characteristics = {
        'n_nodes': graph.numberOfNodes(),
        'n_edges': graph.numberOfEdges(),
        'avg_degree': 0,
        'max_out_degree': 0,
        'max_in_degree': 0,
        'in_heterogeneity': 0,
        'out_heterogeneity': 0,
        'eigenvector_centrality': [0] * N,
        'degree_centrality': [0] * N,
        'betweenness_centrality': [0] * N
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

    # centrality measures
    # eigenvector centrality
    cent = nk.centrality.EigenvectorCentrality(graph)
    stime = time.time()
    cent.run()
    characteristics['eigenvector_centrality'] = cent.ranking()[:N]
    print("Eigenvector Centrality time taken: {} seconds".format(time.time() - stime))

    # Betweenness centrality
    cent = nk.centrality.ApproxBetweenness(graph, epsilon=0.1)
    stime = time.time()
    cent.run()
    characteristics['betweenness_centrality'] = cent.ranking()[:N]
    print("Betweenness Centrality time taken: {} seconds".format(time.time() - stime))

    # Degree centrality
    cent = nk.centrality.DegreeCentrality(graph)
    stime = time.time()
    cent.run()
    characteristics['degree_centrality'] = cent.ranking()[:N]
    print("Degree Centrality time taken: {} seconds".format(time.time() - stime))

    return characteristics


def network_characteristics_gt(graph, sample_size=10000):
    n_samples = 1000
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
    in_degrees = np.array(graph.get_in_degrees(nodes))
    # get out-degree distribution
    out_degrees = np.array(graph.get_out_degrees(nodes))

    characteristics['avg_degree'] = np.average(graph.get_total_degrees(nodes))/2
    characteristics['max_in_degree'] = np.max(in_degrees)
    characteristics['max_out_degree'] = np.max(out_degrees)


    # TODO: NEED TO CHANGE TO REFLECT HOW IT WAS DONE IN PREVIOUS PAPER
    in_degree_stdevs = []
    out_degree_stdevs = []
    in_std_error = []
    out_std_error = []
    in_avg_degree = []
    out_avg_degree = []
    het_in = []
    het_out = []
    for i in range(n_samples):
        in_samples = np.random.choice(in_degrees, sample_size, replace=True)
        out_samples = np.random.choice(out_degrees, sample_size, replace=True)
        in_degree_stdevs.append(np.std(in_samples))
        out_degree_stdevs.append(np.std(out_samples))
        in_std_error.append(sem(in_samples))#in_degree_stdevs[-1] / np.sqrt(1000))
        out_std_error.append(sem(out_samples))#out_degree_stdevs[-1] / np.sqrt(1000))
        in_avg_degree.append(np.average(in_samples))
        out_avg_degree.append(np.average(out_samples))
        het_in.append(np.std(in_samples) / np.average(in_samples))
        het_out.append(np.std(out_samples) / np.average(out_samples))

    #characteristics['in_heterogeneity'] = '{} \pm {} '.format(
    #    np.average(in_degree_stdevs) / (np.average(in_avg_degree)), np.average(in_std_error))
    #characteristics['out_heterogeneity'] = '{} \pm {} '.format(
    #    np.average(out_degree_stdevs) / (np.average(out_avg_degree)), np.average(out_std_error))

    in_error = np.sqrt(np.sum(np.square(np.array(het_in) - np.average(het_in)))/n_samples)
    out_error = np.sqrt(np.sum(np.square(np.array(het_out) - np.average(het_out))) / n_samples)
    characteristics['in_heterogeneity'] = '{} \pm {}'.format(np.average(het_in), in_error)
    characteristics['out_heterogeneity'] = '{} \pm {}'.format(np.average(het_out), out_error)
    return characteristics


def influence_gain(graphs_2016, graphs_2020, N=100):
    #influencers_by_bias_2016 = top_influencers(-1, graphs_2016)
    #influencers_by_bias_2020 = top_influencers(-1, graphs_2020)

    user_to_ci_2016 = {}
    user_to_ci_2020 = {}
    #uid_to_vertex_2016 = {}

    # load user map pickle
    user_map = pickle.load(open('/home/crossb/packaged_ci/maps/user_map_2020.pkl', 'rb'))
    apply_user_map = lambda x: user_map[int(x)]['name']
    verified_map = lambda x: user_map[x]['verified']
    user_stats = []

    for bias in graphs_2016.keys():
        graph_2016 = graphs_2016[bias]
        graph_2020 = graphs_2020[bias]

        bias_stats = []
        bias_2016 = {}
        bias_2020 = {}

        for v in graph_2016.vertices():
            uid = graph_2016.vp.user_id[v]
            user_to_ci_2016[uid] = user_to_ci_2016.get(uid, 0) + graph_2016.vp.CI_out[v]
            bias_2016[uid] = graph_2016.vp.CI_out[v]
            #uid_to_vertex_2016[uid] = v

        for v in graph_2020.vertices():
            uid = graph_2020.vp.user_id[v]
            user_to_ci_2020[uid] = user_to_ci_2020.get(uid, 0) + graph_2020.vp.CI_out[v]
            bias_2020[uid] = graph_2020.vp.CI_out[v]

        for uid in set(bias_2016.keys()) | set(bias_2020.keys()):
            ci_2016 = bias_2016.get(uid, 0)
            ci_2020 = bias_2020.get(uid, 0)
            bias_stats.append({'user_id': uid, 'CI_2016': ci_2016, 'CI_2020': ci_2020})

        bias_stats = pd.DataFrame(bias_stats)
        bias_stats.user_id = bias_stats.user_id.astype(int)
        bias_stats.CI_2016 /= np.max(bias_stats.CI_2016)
        bias_stats.CI_2020 /= np.max(bias_stats.CI_2020)

        bias_stats['delta'] = bias_stats.CI_2020 - bias_stats.CI_2016
        bias_stats = bias_stats[bias_stats.user_id.isin(set(user_map.keys()))]
        bias_stats['user_handle'] = bias_stats.user_id.apply(apply_user_map)
        bias_stats['verified'] = bias_stats.user_id.apply(verified_map)
        bias_stats = bias_stats.sort_values(by='delta', ascending=False)
        bias_stats = bias_stats[(bias_stats.delta >= 0.01) | (bias_stats.delta <= -0.01)]
        plot_influence_gain(bias_stats, N, 'Largest Change in Influence {}'.format(bias),
                            'delta_influence_stem_{}'.format(bias))
        bias_stats.to_csv('results/top_influencers_{}.csv'.format(bias), index=False)



    for uid in set(user_to_ci_2016.keys()) | set(user_to_ci_2020.keys()):
        ci_2016 = user_to_ci_2016.get(uid, 0)
        ci_2020 = user_to_ci_2020.get(uid, 0)
        user_stats.append({'user_id': uid, 'CI_2016': ci_2016, 'CI_2020': ci_2020})

    user_stats = pd.DataFrame(user_stats)
    user_stats.user_id = user_stats.user_id.astype(int)
    user_stats.CI_2016 /= np.max(user_stats.CI_2016)
    user_stats.CI_2020 /= np.max(user_stats.CI_2020)

    user_stats['delta'] = user_stats.CI_2020 - user_stats.CI_2016


    print("len user_stats before", len(user_stats))
    user_stats = user_stats[user_stats.user_id.isin(set(user_map.keys()))]
    print("len user_stats after", len(user_stats))

    user_stats['user_handle'] = user_stats.user_id.apply(apply_user_map)
    user_stats['verified'] = user_stats.user_id.apply(verified_map)
    user_stats = user_stats.sort_values(by='delta', ascending=False)
    user_stats = user_stats[(user_stats.delta >= 0.01)|(user_stats.delta <= -0.01)]
    plot_influence_gain(user_stats, N, 'Largest Change in Influence Combined',
                        'delta_influence_stem_{}'.format('combined'))
    user_stats.to_csv('results/top_influencers.csv', index=False)
    #print("Change in Top {} 2016 user's CI".format(N))
    #for rank, (uid, ci_2020, ci_2016, delta) in enumerate(sorted(delta_CI, key=lambda x: x[1], reverse=True)[:N]):
    #    try:
    #        print("{}. {}: {} - {} = {}".format(rank, user_map[int(uid)], ci_2020, ci_2016,  delta))
    #    except KeyError as e:
    #        pdb.set_trace()
#
    #print("Top {} Users by CI gain".format(N))
    #for rank, (uid, ci_2020, ci_2016, delta) in enumerate(sorted(delta_CI, key=lambda x: x[3], reverse=True)[:N]):
    #    try:
    #        print("{}. {}: {} - {} = {}".format(rank, user_map[int(uid)], ci_2020, ci_2016, delta))
    #    except KeyError as e:
    #        pdb.set_trace()
#
    #print("Top {} users by CI loss".format(N))
    #for rank, (uid, ci_2020, ci_2016, delta) in enumerate(sorted(delta_CI, key=lambda x: x[3], reverse=False)[:N]):
    #    try:
    #        print("{}. {}: {} - {} = {}".format(rank, user_map[int(uid)], ci_2020, ci_2016, delta))
    #    except KeyError as e:
    #        pdb.set_trace()
    return user_stats


def plot_influence_gain(delta_influence_df, N, title='Largest Change in Influence', filename='delta_influence_stem'):
    print("Plotting influence game {}".format(filename))
    #top_bot_ten = delta_influence_df.nlargest(10, 'delta').append(delta_influence_df.nsmallest(10, 'delta'))
    top_ten = delta_influence_df.nlargest(N, 'delta')
    bot_ten = delta_influence_df.nsmallest(N, 'delta')
    # get the top 10 losses and top N gains
    plt.stem(top_ten.user_handle, top_ten.delta, 'g', markerfmt='go', label='Largest CI gains')
    plt.stem(bot_ten.user_handle, bot_ten.delta, 'r', markerfmt='ro', label='Largest CI losses')
    plt.xticks([x for x in top_ten.user_handle]+[x for x in bot_ten.user_handle],
               rotation='vertical', fontsize=10)

    plt.title(title)
    plt.ylabel(r'min-max normalized $\delta$ CI')
    plt.xlabel('User handle')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('results/{}.pdf'.format(filename))
    plt.clf()
    return


def all_network_stats():
    graphs_path = '/home/crossb/packaged_ci/graphs/2020/'
    write_path = '/home/crossb/research/elites_2020/information_diffusion-master/results'
    for path, year in [('/home/crossb/packaged_ci/graphs/2016/', '2016'),
                       ('/home/crossb/packaged_ci/graphs/2020/', '2020')]:
        graphs_gt = load_graphs_gt(path, year)
        network_stats = {}
        for bias, graph in graphs_gt.items():
            samples = 78911 if year == '2016' else 24000
            network_stats[bias] = network_characteristics_gt(graph, samples)

        # write to csv
        with open(os.path.join(write_path, 'network_characteristics_{}.csv'.format(year)), 'w') as outfile:
            write_header = True
            for bias, characteristics in network_stats.items():
                if write_header:
                    print('{},{},{},{},{},{},{},{}'.format('', *characteristics.keys()), file=outfile)
                    write_header = False
                print(','.join([bias] + [str(value) for stat, value in characteristics.items()]), file=outfile)
                #print("{} Network Statistics".format(bias))
                #for stat, value in characteristics.items():
                #    print("{}: {}".format(stat, value))
                #print("===========================================")




    # perform some centrality analysis
    #graphs = load_graphs_nk(graphs_path)
    #nk_stats = {}
    #N = 30
    #for bias, graph in graphs.items():
    #    if bias == 'left extreme':
    #        continue
    #    print("Gathering network stats for", bias)
    #    nk_stats[bias] = network_characteristics_nk(graph, N)

    #for bias, characteristics in nk_stats.items():
    #    print("{} Network Statistics".format(bias))
    #    for stat, value in characteristics.items():
    #        if "centrality" in stat:
    #            print("{}: {}".format(
    #                stat,
    #                ','.join(['(Node: {}: {})'.format(graphs_gt[bias].vp.user_id[n], v) for (n, v) in value])))
    #        else:
    #            print("{}: {}".format(stat, value))
    #    print("===========================================")
    return


def shortest_paths_to_target(graph, target):
    if target is None:
        return

    uid_to_shortest_paths = {}
    for v in graph.vertices():
        shortest_paths = gt.all_shortest_paths(graph, v, target)
        uid_to_shortest_paths[graph.vp.user_id[v]] = [x for x in shortest_paths]

    return uid_to_shortest_paths


def get_shortest_paths(graph, source, target):
    shortest_paths = [x for x in gt.all_shortest_paths(graph, source, target)]
    return {
        'user_id': graph.vp.user_id[target],
        'num_shortest_paths': len(shortest_paths),
        'len_shortest_path': len(shortest_paths[0]) if len(shortest_paths) > 0 else 0
    }


def parallel_shortest_paths_from_source(graph, source, threads):
    #pdb.set_trace()
    with Pool(threads) as pool:
        sp_args = [(graph, source, int(v)) for v in graph.vertices()]
        paths_data = pool.starmap(get_shortest_paths, sp_args)
    return paths_data


def shortest_paths_from_source(graph, source):
    if source is None:
        return

    uid_to_shortest_paths = {}
    for v in graph.vertices():
        shortest_paths = gt.all_shortest_paths(graph, source, v)
        uid_to_shortest_paths[graph.vp.user_id[v]] = [x for x in shortest_paths]

    return uid_to_shortest_paths


def anti_trump(graphs, threads):
    #pdb.set_trace()
    stime = time.time()
    for bias, graph in graphs.items():
        # get trump vertex
        trump_v = None
        for v in graph.vertices():
            if graph.vp.user_id[v] == '25073877':
                trump_v = int(v)

        if trump_v is None:
            continue
        #user_shortest_paths = parallel_shortest_paths_from_source(graph, trump_v, threads)
        user_shortest_paths = pd.DataFrame(parallel_shortest_paths_from_source(graph, trump_v, threads))
        #user_shortest_path_lengths = pd.DataFrame(
        #    [{'user_id': uid, 'shortest_path_length': len(paths)} for uid, paths in user_shortest_paths.items()])
        user_shortest_paths.to_csv('results/{}_user_to_n_shortest_paths.csv'.format(bias))

        #user_shortest_path_lengths = {uid: len(paths) for uid, paths in user_shortest_paths.items()}
    print('Anti-Trump time taken: {} seconds'.format(time.time() - stime))
    return


def read_edgeslist(path, columns=('auth_id', 'infl_id')):
    edges = dd.read_csv(path, delimiter=',',
                        usecols=columns,
                        dtype={col: np.int64 for col in columns}
                        )
    return edges


def engagement(graph, top_n):
    """
    The idea behind engagement is to find out which influencers and which biases are attracting
    the most devoted / vocal followers.

    Method:
        1) Loop over nodes, get unique out tweets and out users
        2) For each out tweet, get fraction of total out users retweeted that tweet, keep track of bias here as well.

    :return:
    """

    for node in graph.vertices():

    return


def main():
    print("Default number of threads:", nk.getCurrentNumberOfThreads())
    num_threads = 16
    nk.setNumberOfThreads(64)
    print("Updated number of threads:", nk.getCurrentNumberOfThreads())
    #graph = load_from_graphtool('/home/crossb/packaged_ci/graphs/2020/Right_news_2020_ci.gt')
    graphs_2020 = load_graphs_gt('/home/crossb/packaged_ci/graphs/2020/', year='2020')
    graphs_2016 = load_graphs_gt('/home/crossb/packaged_ci/graphs/2016/', year='2016')
    #anti_trump(graphs_2020, num_threads)

    delta_df = influence_gain(graphs_2016, graphs_2020, N=20)
    #influencer_network_anlaysis(year=2016)
    #influencer_network_anlaysis(year=2020)
    #all_network_stats()
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