from multiprocessing import Pool
from two_agents import tabular_ql as ql
import random, math, numpy
import weka.core.jvm as jvm
from weka.core.converters import Loader
import weka.core.converters as converters
from weka.core.dataset import Instances, Attribute, Instance
from weka.classifiers import Classifier

classifier_name = "weka.classifiers.trees.J48"


class NewCCActionSelector:

    def __init__(self, action_space, state_length, state_weights, cc_input_file, agent_input_file):
        jvm.start()
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(agent_input_file)  # data/data_agent1_wPub.arff")
        data.class_is_last()
        nA = int(max(action_space)) + 1
        cls = Classifier(classname=classifier_name)
        cls.build_classifier(data)
        ccmap = load_coord_confidence(cc_input_file)


def load_coord_confidence(fname):
    from ast import literal_eval as make_tuple
    import csv
    out_map = {}
    inf = open(fname, 'r')
    inp = csv.reader(inf, delimiter=':')
    for row in inp:
        t = 1 - (float(row[1]))  # math.exp(float(row[1]))-1.0
        # if t < 0.0001:
        #    t=0.0001
        out_map[make_tuple(row[0])] = t  # float(row[1])
    inf.close()
    return out_map


def combine_decisions2(chat_iA, cc_iA, chat_conf,
                       cc_conf):  # return -1 if no recommendation, otherwise id of recom action
    if (chat_conf < CONF_THRESHOLD) and (cc_conf < CONF_THRESHOLD):
        return -1, "None"
    else:
        if cc_conf >= CONF_THRESHOLD:
            return cc_iA, "CC"
        elif chat_conf >= CONF_THRESHOLD:
            return chat_iA, "CHAT"


def combine_decisions1(chat_iA, cc_iA, chat_conf,
                       cc_conf):  # return -1 if no recommendation, otherwise id of recom action
    if (cc_conf >= CONF_THRESHOLD):
        return cc_iA, "CC"
    else:
        return -1, "None"


def dist(x, y):
    assert len(x) == len(y)
    total = 0.0
    for i in range(len(x)):
        total += (x[i] - y[i]) * (x[i] - y[i])
    return math.sqrt(total)


def exactMatch(key1, key2, indices):
    assert len(key1) == len(key2)
    for i in indices:
        if key1[i] != key2[i]:
            return False
    return True


def approxMatch(key1, key2, indices):
    assert len(key1) == len(key2)
    k1, k2 = [key1[i] for i in indices], [key2[i] for i in indices]
    return dist(k1, k2)


def matchKey(a_id, obs, map_obs):  # map_obs is a map[key]->something; which key most closely matches obs?
    dist_map = {}
    if a_id == 1:
        if obs[3] == 0 and obs[4] == 0:
            exact = [10]
            approx = range(10)
        else:
            exact = [1, 3, 4, 6, 10]  # indices to be matched exactly
            approx = [0, 2, 5, 7, 8, 9]
    else:  # a_id==2
        exact = [2]
        approx = range(2)  # indices to use for distance calculation

    for key in map_obs:
        if exactMatch(key, obs, exact):
            dist_map[key] = approxMatch(key, obs, approx)
        else:
            dist_map[key] = 100000.0
    ret_key = min(dist_map, key=dist_map.get)
    ret_list = []
    if dist_map[ret_key] >= 5.0:
        return ret_list  # no good match
    else:
        min_dist = dist_map[ret_key]
        for key in dist_map:
            if dist_map[key] == min_dist:
                ret_list.append(key)
        return ret_list