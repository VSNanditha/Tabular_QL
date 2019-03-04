import weka.core.jvm as jvm  # <=== this is meant for the client
from weka.classifiers import Classifier
# import weka from /Users/sneha/anaconda3/lib/python3.6/site-packages/
from weka.core.converters import Loader
from weka.core.dataset import Instance

# import wekaexamples.helper as helper
# from weka.core.converters import Loader


# from weka.filters import Filter
# from weka.core.classes import Random, from_commandline
# import weka.plot.classifiers as plot_cls
# import weka.plot.graph as plot_graph
# import weka.core.typeconv as typeconv

# ===============CHANGE==============================================
# action_space = [0,1]
# arff = "arff/symmetric_demo_bdn_noise_agent_v1.arff"
# classifier_name = "weka.classifiers.trees.J48"  # Decision-tree
classifier_name = 'weka.classifiers.trees.RandomForest'
# ===================================================================


# ===================================================================
# Assumes state is a vector of length 'state_length', and
# has numerical elements. Also assumes that the client will call
# jvm.start() before instantiating, and jvm.stop() after finishing.
# action_space is like so: [0,1,2,3,4] (for Discrete(5))
# ===================================================================

class CHATActionSelector(object):

    def __init__(self, action_space, state_length, arff_fname):
        self.action_space = action_space
        jvm.start()
        loader = Loader(classname="weka.core.converters.ArffLoader")
        data = loader.load_file(arff_fname)
        data.class_is_last()
        self.cls = Classifier(classname="weka.classifiers.trees.RandomForest")
        self.cls.build_classifier(data)
        dummy_state = [0.0]*state_length
        self.instance = Instance.create_instance(dummy_state)
        self.instance.dataset = data

    def select_action(self, state, threshold):
        for i in range(len(state)):
            self.instance.set_value(i, state[i])
        distribution = list(self.cls.distribution_for_instance(self.instance))
        # print(distribution)
        if max(distribution) >= threshold:
            action_index = distribution.index(max(distribution))
            return int(self.action_space[action_index])
        else:
            return -1

# CHATActionSelector = CHATActionSelector([0, 1, 2, 3, 4, 5, 6, 7], 11, "agent1.arff")
# obs = (1,1,0,0,0,0,1,0,0,0,0)
# print(CHATActionSelector.select_action(obs, 0.7))
