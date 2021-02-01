# Author: Alexandre Bovet <alexandre.bovet (at) uclouvain.be>, 2018
#
# License: GNU General Public License v3.0

import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


import pandas as pd
import numpy as np

import pdb
import time
import pickle
import graph_tool.all as gt


periods = ['june-nov']
year = '2020'

save_dir = '/home/crossb/packaged_ci/data/{}/urls/'.format(year)
network_dir = '/home/crossb/packaged_ci/graphs/{}/'.format(year)

update_res = False
#raise Exception

#%% load results (for updating results)

if update_res:
    with open(os.path.join(save_dir, 'influencer_rankings_simple.pickle'), 'rb') as fopen:
        
        ranks_dict_simple = pickle.load(fopen)
    
    with open(os.path.join(save_dir, 'influencer_rankings_complete.pickle'), 'rb') as fopen:
        
        ranks_dict_complete = pickle.load(fopen)

#%% print top ranked for each graph
    
def user_id_to_vertex_id(G, uid):
    
    if isinstance(uid, list):
    
        return [np.where(G.vp.user_id == u)[0][0] for u in uid]
    
    else:
        return np.where(G.vp.user_id == uid)[0][0]
    

def return_ranking(a):
    b = np.zeros_like(a, dtype=int)
    sorting = np.argsort(a)[::-1]
    for i in range(a.shape[0]):
        b[sorting[i]] = i
    return b


def uids_to_handles():
    return


def get_topranked_userids_CI(G, rank='CI_out',
                          topnum=200):
    ranker = G.vp[rank].a
    sortind = ranker.argsort()
    
    if G.vp.user_id.value_type() == 'python::object' or G.vp.user_id.a is None:
        user_ida = np.array([G.vp.user_id[v] for v in G.vertices()], dtype=np.int64)
        user_ids = user_ida[sortind[::-1]][:topnum]
    else:
        user_ids = G.vp.user_id.a[sortind[::-1]][:topnum]
        
    top_v = G.get_vertices()[sortind[::-1]][:topnum]
    
    if 'most_used_client_in' in G.vp:
        most_used_clients_in = [G.vp.most_used_client_in[v] for v in top_v]
    if 'most_used_client_out' in G.vp:
        most_used_clients_out = [G.vp.most_used_client_out[v] for v in top_v]

        
    uid_names = []
    uid_verified =[]
    print('retrieving screennames')
    # load user map pickle
    user_map = pickle.load(open('/home/crossb/packaged_ci/maps/user_map_{}.pkl'.format(year), 'rb'))
    res = []
    #uid_to_user = lambda x: ('@' + user_map[x]['name'], user_map[x]['verified'])
    #res = [uid_to_user(x) for x in user_ids]
    for uid in user_ids:
        if uid in user_map:
            if year == '2016':
                verified = False
            else:
                verified = user_map[uid]['verified']
            res.append(('@' + user_map[uid]['name'], verified))
        else:
            print(uid)
            res.append(('@???????', False))

    for uname, uverif in res:
        uid_names.append(uname)
        uid_verified.append(uverif)

    #return [(names, verif, uid, kout, kin, pr, CI_out, CI_in,
    #         kcore, eig_rev, katz_rev, k_out_rank, muci, muco, osri, osro) for\
    #         names, verif, uid, kout, kin, pr, CI_out, CI_in,
    #         kcore, eig_rev, katz_rev, k_out_rank, muci, muco, osri, osro
    return [(names, verif, uid, kout, kin, CI_out, CI_in, katz_rev, k_out_rank) for \
            names, verif, uid, kout, kin, CI_out, CI_in, katz_rev, k_out_rank
              in zip(uid_names,
                     uid_verified,
                     user_ids,
         G.vp.k_out.a[sortind[::-1]][:topnum],
         G.vp.k_in.a[sortind[::-1]][:topnum],
         return_ranking(G.vp.CI_out.a)[sortind[::-1]][:topnum],
         return_ranking(G.vp.CI_in.a)[sortind[::-1]][:topnum],
         return_ranking(G.vp.katz_rev.a)[sortind[::-1]][:topnum],
         return_ranking(G.vp.k_out.a)[sortind[::-1]][:topnum])]#,
         #most_used_clients_in,
         #most_used_clients_out)]
    
    
#%% for CI
    
columns = ['screenname', 
           'is_verified',
           'user_id',
           '$k_\mathrm{out}$', 
           '$k_\mathrm{in}$', 
             '$\mathrm{CI_{out}}$', 
             '$\mathrm{CI_{in}}$',             
             'katz',
             'HD']

if not update_res:
    ranks_dict_complete = dict()
    ranks_dict_simple = dict()

t0 = time.time()

#media_types = ['fake', 'right_extreme', 'right', 'right_leaning',
#                   'center', 'left_leaning', 'left', 'left_extreme']
media_types = {
    '2020': ['fake', 'right_extreme', 'right', 'right_leaning',
             'center', 'left_leaning', 'left', 'left_extreme'],
    '2016': ['fake', 'extreme_bias_right', 'right', 'lean_right',
             'center', 'lean_left', 'left', 'extreme_bias_left']
               }

for media_type in media_types[year]:
    if True:
        ranks_dict_complete[media_type] = dict()
        ranks_dict_simple[media_type] = dict()
    else:
        if media_type not in ranks_dict_complete.keys():
            ranks_dict_complete[media_type] = dict()
        if media_type not in ranks_dict_simple.keys():
            ranks_dict_simple[media_type] = dict()
        
    #for period in periods:
    #    if not update_res:
    #        ranks_dict_complete[media_type][period] = dict()
    #        ranks_dict_simple[media_type][period] = dict()
    #    else:
    #        if period not in ranks_dict_complete[media_type].keys():
    #            ranks_dict_complete[media_type][period] = dict()
    #        if period not in ranks_dict_simple[media_type].keys():
    #            ranks_dict_simple[media_type][period] = dict()


    print('loading graphs')

    print(media_type)
    #print(period)

    #Gcomplete = gt.load_graph(os.path.join(save_dir, 'retweet_graph_' + \
    #      media_type + '_complete_' + period + '.gt'))
    Gsimple = gt.load_graph(os.path.join(network_dir, media_type + '_' + year +'_simple_ci.gt'))

    #for ranker in ['pagerank_rev', 'CI_out', 'CI_in', 'eigenvec', 'katz','k_out']:
    for ranker in ['CI_out', 'CI_in', 'katz']:
        print(ranker)


        #ranks_dict_complete[media_type][period][ranker] = \
        #pd.DataFrame(data=get_topranked_userids_CI(Gcomplete,
        #         rank=ranker, topnum=200), columns=columns)
        ranks_dict_simple[media_type][ranker] = \
        pd.DataFrame(data=get_topranked_userids_CI(Gsimple,
                 rank=ranker, topnum=200), columns=columns)

        print(time.time()-t0)
#%% save results
with open(os.path.join(save_dir, 'influencer_rankings_simple_{}.pickle'.format(year)), 'wb') as fopen:
    
    pickle.dump(ranks_dict_simple, fopen)

#with open(os.path.join(save_dir, 'influencer_rankings_complete.pickle'), 'wb') as fopen:
#
#    pickle.dump(ranks_dict_complete, fopen)


