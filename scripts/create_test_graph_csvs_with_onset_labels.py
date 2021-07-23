import numpy as np
import itertools
import partitura
from math import floor
import networkx as nx
from scipy.sparse import csr_matrix 
import matplotlib.pyplot as plt
import dgl, torch
import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from trajectory import buildTrajectory, weightsOfTrajPoints_Normalized, SetOfPoints

dirname = os.path.dirname(__file__)
par = lambda x : os.path.abspath(os.path.join(x, os.pardir))
par_directory = os.path.join(par(par(par(dirname))), "mozart_piano_sonatas", "utils")
sys.path.append(par_directory)

from feature_matrices import load_tsv

def add_edge(x, y, etype, gd, nd):
    k = ("note", etype, "note")
    nl = nd[x["id"]]
    nr = nd[y["id"]]
    if k in gd.keys():
        
        l = gd[k][0] + [nl]
        
        r = gd[k][1] + [nr]
        gd[k] = (l, r)
    else : 
        gd[k] = ([nl], [nr])
        
    return gd

# def simultaneous_edge(x, y):
#     return (x, "simultaneous", y)

# def while_edge(x, y):
#     return (x, "while", y)

# def rest_edge_r(x):
#     return (x, "rest", 128)

# def rest_edge_l(x):
#     return (128, "rest", x)



def graph_dict_from_na(na, cad_onsets):
    '''Group note_array list by time, effectively to chords.
    
    Parameters
    ----------
    note_array : array(N, 5)
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    cad_onsets : list
        The onset positions of cadences for the corresponding note array
    t_sig : list
        A list of time signature in the piece.
    Returns
    -------
    chords : list(list(tuples))
    '''
    uniques = np.unique(na["id"])
    if uniques == ["None"]:
        na["id"] = np.array(range(len(na)))
        uniques = na["id"]
    node_dict = dict()
    graph_dict = dict()
    node_dict = {idx : i for i, u in enumerate(uniques) for idx in na["id"][np.where(na["id"] == u)] }
    message_dict = {i : na[np.where(na["id"] == idx)]["pitch"][0] for idx, i in node_dict.items()}
    cad_dict = {i : True in np.isclose(na[np.where(na["id"] == idx)]["onset_beat"][0], cad_onsets) for idx, i in node_dict.items()}
    for x in na:
        sim = na[np.where((np.isclose(na["onset_beat"], x["onset_beat"]) == True) & (na["pitch"] != x["pitch"] ))]
        for y in sim:
            graph_dict = add_edge(x, y, "same time as", graph_dict, node_dict)
        cons = na[np.where(np.isclose(na["onset_beat"], x["onset_beat"]+x["duration_beat"]) == True)]
        for y in cons:
            graph_dict = add_edge(x, y, "followed by", graph_dict, node_dict)
        inv_cons = na[np.where(np.isclose(na["onset_beat"]+na["duration_beat"], x["onset_beat"]) == True)]
        for y in inv_cons:
            graph_dict = add_edge(x, y, "follows", graph_dict, node_dict)
        dur = na[np.where((x["onset_beat"] < na["onset_beat"]) & (x["onset_beat"]+x["duration_beat"] > na["onset_beat"]) ) ]
        for y in dur:
            graph_dict = add_edge(x, y, "plays through", graph_dict, node_dict)
        # Time signature edge
    message_mask = [message_dict[i] for i in sorted(list(message_dict.keys())) ]
    node_mask = [cad_dict[i] for i in sorted(list(cad_dict.keys()))]

    return graph_dict, message_mask, node_mask


def select_ts(x, t_sig):
    for y in t_sig:
        if (x["onset_beat"] < y["end_beat"] or y["end_beat"] == -1) and x["onset_beat"] >= y["onset_beat"]:
            return y["nominator"]
    print(t_sig, x)

def graph_csv_from_na(na, ra, t_sig):
    '''Turn note_array to heterogeneous graph dictionary.
    
    Parameters
    ----------
    na : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    ra : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.

    '''
    note = pd.DataFrame(
        {
            "nid" : np.array(range(len(na))),
            "pitch" : na["pitch"],
            "onset" : na["onset_beat"],
            "duration" : na["duration_beat"],
            "ts" : np.array(list(map(lambda x : select_ts(x, t_sig), na))),
            "label" : np.array([1 if x["onset_beat"]%select_ts(x, t_sig) == 0 else 0 for x in na])
        }
    )

    # ts = pd.DataFrame(
    #     {
    #         "start"       : t_sig["onset_beat"],
    #         "finish"      : t_sig["end_beat"],
    #         "numerator"   : t_sig["nominator"],
    #         "denominator" : t_sig["denominator"]
    #     }
    # )
    rest_ends = list(set([ n["onset_beat"] for n in na if np.where(na["onset_beat"] + na["duration_beat"] == n["onset_beat"])[0].size == 0 ]))
    note_ends = na["onset_beat"] + na["duration_beat"]
    re = list()
    for r in ra:
        u = np.where(((na["onset_beat"] >= r["onset_beat"]) &  (na["onset_beat"] < r["onset_beat"]+r["duration_beat"])))[0]
        v = np.where(((na["onset_beat"] < r["onset_beat"]) &  (na["onset_beat"]+na["duration_beat"] > r["onset_beat"])))[0]
        if u.size == 0 and v.size==0:
            re.append(r)
    re = np.unique(np.array(re, dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4')]))

    sim_src = list() 
    sim_des = list()
    cons_src = list()
    cons_des = list()
    dur_src = list()
    dur_des = list()
    cons_nr_src = list()
    cons_nr_des = list()
    cons_rn_src = list()
    cons_rn_des = list()
    # ts_src = list()
    # ts_des = list()

    for i, x in enumerate(na):
        for j in np.where((np.isclose(na["onset_beat"], x["onset_beat"]) == True) & (na["pitch"] != x["pitch"]))[0]:
            sim_src.append(i)
            sim_des.append(j)
        
        for j in np.where(np.isclose(na["onset_beat"], x["onset_beat"]+x["duration_beat"]) == True)[0]:
            cons_src.append(i)
            cons_des.append(j)

        if re.size > 0:
            for j in np.where(np.isclose(re["onset_beat"], x["onset_beat"]+x["duration_beat"]) == True)[0]:
                cons_rn_src.append(i)
                cons_rn_des.append(j)
            
        for j in np.where((x["onset_beat"] < na["onset_beat"]) & (x["onset_beat"]+x["duration_beat"] > na["onset_beat"]) )[0]:
            dur_src.append(i)
            dur_des.append(j)

        # # Time Signature
        # for y in t_sig:
        #     if (y["nominator"], y["denominator"]) in ts_voc.values():
        #         for k, v in ts_voc.items():
        #             if v == (y["nominator"], y["denominator"]):
        #                 break
        #     elif not ts_voc:
        #         k = 0
        #         ts_voc[k] = (y["nominator"], y["denominator"])
        #     else :
        #         k = max(ts_voc.keys())+1
        #         ts_voc[k] = (y["nominator"], y["denominator"])

        #     if (x["onset_beat"] < y["end_beat"] or y["end_beat"] == -1) and x["onset_beat"] >= y["onset_beat"]:
        #         ts_src.append(k)
        #         ts_des.append(i)


    
    if re.size > 0:
        for i, r in enumerate(re):
            for j in np.where(np.isclose(na["onset_beat"], r["onset_beat"]+r["duration_beat"]) == True)[0]:
                cons_nr_src.append(i)
                cons_nr_des.append(j)

        rest = pd.DataFrame(   
            {
                "rid" : np.array(range(len(re))),
                "onset" : re["onset_beat"],
                "duration" : re["duration_beat"],
                "ts" : np.array(list(map(lambda x : select_ts(x, t_sig), re))),
                "label" : np.zeros(len(re))
            }
        )
    else : 
        rest = pd.DataFrame()
    sim = pd.DataFrame(
        {
            "src" : sim_src,
            "des" : sim_des
        }
    )
    cons = pd.DataFrame(
        {
            "src" : cons_src,
            "des" : cons_des,
        }
    )
    dur = pd.DataFrame(
        {
            "src" : dur_src,
            "des" : dur_des
        }
    )
    cons_nr = pd.DataFrame(
        {
            "src" : cons_nr_src,
            "des" : cons_nr_des,
        }
    )
    cons_rn = pd.DataFrame(
        {
            "src" : cons_rn_src,
            "des" : cons_rn_des,
        }
    )
    # time_signature = pd.DataFrame(
    #     {
    #         "src" : cons_rn_src,
    #         "des" : cons_rn_des,
    #     }
    # )
    return note, rest, sim, cons, dur, cons_nr, cons_rn



def data_loading(tsv_dir, score_dir):
    """
    Create a Trainset from annotations and scores.

    Parameters
    ----------
    tsv_dir : path
        Path for tsv file with annotations.
    score_dir : path
        Path for folder with scores.
    Returns
    -------
    scores : dict
        Keys are piece names, i.e. K279-1. 
        Values are individual score directories.
    annotations : dataframe
        Read from tsv file with annotations.
    """
    annotations = load_tsv(tsv_dir, stringtype=False)
    scores = dict()
    for score_name in os.listdir(score_dir):
        if score_name.endswith(".musicxml"):
            key = os.path.splitext(score_name)[0]
            scores[key] = os.path.join(score_dir, score_name)       
    return scores, annotations



def filter_cadences_from_annotations(annotations):
    """
    Create a Trainset from annotations and scores.

    Parameters
    ----------
    annotations : dataframe
        Read from tsv file with annotations.
    Returns
    -------
    phrase_dict : dictionary
        Keys are piece names, i.e. K279-1. 
        Values are lists of floats with the beat positions where cadences occur.
    """
    
    annotations["cad_pos"] = annotations["timesig"].astype(str).str[0].astype(int)*(annotations["mc"]-1) + annotations["onset"].astype(float)*annotations["timesig"].astype(str).str[0].astype(int) 
    annotations["filename"] = annotations.index.get_level_values("filename")
    cad_dict = dict()
    for filename in annotations.filename.unique():
        cad_dict[filename] = list(zip(annotations.loc[annotations["filename"] == filename, "cad_pos"].to_list(), annotations.loc[annotations["filename"] == filename, "cadence"].to_list()))
    # for index in pe.shape[0]:
    #   phrase_dict[phrase_end.iloc[index]["filename"]] :
    return cad_dict


def filter_ts_end(ts, part):
    if ts.end:
        return part.beat_map(ts.end.t)
    else:
        return -1

def create_data(tsv_dir, score_dir, save_dir=None):
    """
    Create a Trainset from annotations and scores.

    Parameters
    ----------
    phrase_dict : dict
        Keys are piece names, i.e. K279-1. 
        Values are lists of floats with the beat positions where end of phrases occurs
    scores : dict
        Keys are piece names, i.e. K279-1. 
        Values are score directories.
    Returns
    -------
    trainset : list of couples
        Every element is a couple with the first element being a graph and the second its label.
    """
    if not save_dir:
        save_dir = score_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    scores, annotations = data_loading(tsv_dir, score_dir)
    phrase_dict = filter_cadences_from_annotations(annotations)
    ts_voc = dict()
    for key, fn in scores.items():
        print(key)
        part = partitura.load_musicxml(fn) 
        
        if isinstance(part,  partitura.score.PartGroup):
            rest_array = np.array([(p.beat_map(r.start.t), p.beat_map(r.duration)) for p in partitura.score.iter_parts(part) for r in p.iter_all(partitura.score.Rest) ], dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4')])
            time_signature = np.array([(p.beat_map(ts.start.t), filter_ts_end(ts, p), ts.beats, ts.beat_type) for p in partitura.score.iter_parts(part) for ts in p.iter_all(partitura.score.TimeSignature) ], dtype=[('onset_beat', '<f4'), ('end_beat', '<f4'), ("nominator", "<i4"),("denominator", "<i4")]) 
        elif isinstance(part, list):
            rest_array = np.array([(p.beat_map(r.start.t), p.beat_map(r.duration)) for p in part for r in p.iter_all(partitura.score.Rest) ], dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4')])
            time_signature = np.array([(p.beat_map(ts.start.t), filter_ts_end(ts, p), ts.beats, ts.beat_type) for p in part for ts in p.iter_all(partitura.score.TimeSignature) ], dtype=[('onset_beat', '<f4'), ('end_beat', '<f4'), ("nominator", "<i4"),("denominator", "<i4")]) 
        else:    
            rest_array = np.array([(part.beat_map(r.start.t), part.beat_map(r.duration)) for r in part.iter_all(partitura.score.Rest)], dtype=[('onset_beat', '<f4'), ('duration_beat', '<f4')])        
            time_signature = np.array(
                [
                    (part.beat_map(ts.start.t), filter_ts_end(ts, part), ts.beats, ts.beat_type) for ts in part.iter_all(partitura.score.TimeSignature)
                ], 
                dtype=[('onset_beat', '<f4'), ('end_beat', '<f4'), ("nominator", "<i4"),("denominator", "<i4")]
            ) 
        note_array = partitura.utils.ensure_notearray(part) 
        # cad_pos, cad_label = zip(*phrase_dict[key])
        note, rest, sim, cons, dur, cons_rn, cons_nr = graph_csv_from_na(note_array, rest_array, time_signature)
        if not os.path.exists(os.path.join(save_dir, key)):
                os.makedirs(os.path.join(save_dir, key))
        note.to_csv(os.path.join(save_dir, key, "note.csv"))
        rest.to_csv(os.path.join(save_dir, key, "rest.csv"))
        # ts.to_csv(os.path.join(save_dir, key, "ts.csv"))
        sim.to_csv(os.path.join(save_dir, key, "note-onset-note.csv"))
        cons.to_csv(os.path.join(save_dir, key, "note-follows-note.csv"))
        dur.to_csv(os.path.join(save_dir, key, "note-during-note.csv"))
        cons_nr.to_csv(os.path.join(save_dir, key, "note-follows-rest.csv"))
        cons_rn.to_csv(os.path.join(save_dir, key, "rest-follows-note.csv"))
        # tse.to_csv(os.path.join(save_dir, key, "ts-connects-note.csv"))


if __name__ == "__main__":
    
    import os
    import pickle

    import pandas as pd
    import pickle
    # from utils import MyGraphDataset
    # from sklearn.preprocessing import LabelEncoder

    dirname = os.path.dirname(__file__)
    par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
    par_dir = par(par(dirname))    
    tsv_dir = os.path.join(par(par_dir), "mozart_piano_sonatas", "formatted", "-C_cadences.tsv")
    score_dir = os.path.join(par_dir, "samples", "xml", "mozart_piano_sonatas")
    save_dir = os.path.join(par(par_dir), "tonnetzcad", "node_classification", "mps_ts_att_onlab")
    data = create_data(tsv_dir, score_dir, save_dir)
    # fn = os.path.join(par_dir, "artifacts", "data", "cadences", "score_graphs.pkl")
    # with open(fn, "wb") as f:
    #     pickle.dump(data, f)
