#Changed to omit fallback to CHAT when conf too low.
#Changed to work with Gym env.
from multiprocessing import Pool
from two_agents import tabular_ql as ql
import random, math, numpy, gym
#from gym.spaces import Tuple

#==========================CHANGE==================================================================
#from domains import FurnitureMovers as gm
#from domains import BlockDudes as gm
#from domains import GuidedNavigationDynamic

gm = gym.make('GuideDog-v1')
num_trials = 8
action_noise = 0.1

#act1 = ['4', '2', '9', '8', '5', '1', '3', '7', '6', '0'] #FM
#act2 = ['4', '2', '9', '8', '5', '1', '3', '7', '6', '0'] #FM

act1 = ['0', '1', '2', '3', '4', '5', '6'] #GNdynamic
act2 = ['0', '1', '2', '3']                #GNdynamic

#act1 = ['2', '3', '4', '1', '0'] #BD
#act2 = ['1', '3', '2', '4', '0'] #BD

#act1 = ['2', '4', '1', '0', '3'] #BDN
#act2 = ['3', '1', '4', '2', '0'] #BDN

#act1 = ['2', '3', '1', '4', '0'] #BDN-symm
#act2 = ['2', '3', '1', '4', '0'] #BDN-symm

#arff1 = "arff/asymmetric_demo_bdn_noise_agent1_v1.arff"
#arff2 = "arff/asymmetric_demo_bdn_noise_agent2_v1.arff"

#arff1 = "arff/symmetric_demo_bdn_noise_agent_v1.arff" #BDN-symm
#arff2 = "arff/symmetric_demo_bdn_noise_agent_v1.arff" #BDN-symm

#arff1 = "arff/symmetric_demo_fm_noise_agent_v1.arff"
#arff2 = "arff/symmetric_demo_fm_noise_agent_v1.arff"

arff1 = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/demonstrations/final_arff/moving_obstacles/agent1.arff"
arff2 = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/demonstrations/final_arff/moving_obstacles/agent2.arff"


classifier_name = "weka.classifiers.trees.J48"
#classifier_name = "weka.classifiers.functions.MultilayerPerceptron"

#options_NN = ["-H", "20"] #NN

cc1file = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/demonstrations/final_arff/moving_obstacles/blind_coord_a1.dat"
cc2file = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/demonstrations/final_arff/moving_obstacles/blind_coord_a2.dat"
CONF_THRESHOLD = 0.7
nF1, nF2 = 11, 3
#==================================================================================================

def load_coord_confidence(fname):
    from ast import literal_eval as make_tuple
    import csv
    out_map = {}
    inf = open(fname,'r')
    inp = csv.reader(inf, delimiter=':')
    for row in inp:
        t = 1-(float(row[1])) #math.exp(float(row[1]))-1.0
        #if t < 0.0001:
        #    t=0.0001
        out_map[ make_tuple( row[0] ) ] = t #float(row[1])
    inf.close()
    return out_map


def combine_decisions2(chat_iA, cc_iA, chat_conf, cc_conf): #return -1 if no recommendation, otherwise id of recom action
    if (chat_conf < CONF_THRESHOLD) and (cc_conf < CONF_THRESHOLD):
        return -1, "None"
    else:
        if cc_conf >= CONF_THRESHOLD:
            return cc_iA, "CC"
        elif chat_conf >= CONF_THRESHOLD:
            return chat_iA, "CHAT"

def combine_decisions1(chat_iA, cc_iA, chat_conf, cc_conf): #return -1 if no recommendation, otherwise id of recom action
    if (cc_conf >= CONF_THRESHOLD):
        return cc_iA, "CC"
    else:
        return -1, "None"
    

def dist(x,y):
    assert len(x)==len(y)
    total = 0.0
    for i in range(len(x)):
        total += (x[i] - y[i]) * (x[i] - y[i])
    return math.sqrt( total )

def exactMatch(key1, key2, indices):
    assert len(key1)==len(key2)
    for i in indices:
        if key1[i] != key2[i]:
            return False
    return True

def approxMatch(key1, key2, indices):
    assert len(key1)==len(key2)
    k1, k2 = [key1[i] for i in indices], [key2[i] for i in indices]
    return dist(k1,k2)
    
def matchKey( a_id, obs, map_obs ): #map_obs is a map[key]->something; which key most closely matches obs?
    dist_map = {}
    if a_id==1:
        if obs[3]==0 and obs[4]==0:
            exact = [10]
            approx = range(10)
        else:
            exact = [1,3,4,6,10] #indices to be matched exactly
            approx = [0,2,5,7,8,9]
    else: # a_id==2
        exact=[2]
        approx = range(2) #indices to use for distance calculation
            

    for key in map_obs:
        if exactMatch(key, obs, exact):
            dist_map[ key ] = approxMatch( key, obs, approx )
        else:
            dist_map[ key ] = 100000.0
    ret_key = min( dist_map, key=dist_map.get )
    ret_list = []
    if dist_map[ret_key] >= 5.0:
        return ret_list  # no good match
    else:
        min_dist = dist_map[ ret_key ]
        for key in dist_map:
            if dist_map[key] == min_dist:
                ret_list.append( key )
        return ret_list
        
# for trial in range(num_trials):


def do_thread(trip):  # trial, hat_decay, alpha ):
    trial, hat_decay, alpha = trip[0], trip[1], trip[2]
    print (trial, hat_decay, alpha)
    # return
    
    import weka.core.jvm as jvm
    from weka.core.converters import Loader
    import weka.core.converters as converters
    from weka.core.dataset import Instances, Attribute, Instance
    from weka.classifiers import Classifier

    random.seed( trial )
    
    num_episodes = 100000
    max_num_steps = 200
    SKIP_EPISODES = 100
    
    prob_HAT = 1.0
    prob_HAT_decay = hat_decay #math.exp(math.log(0.05)/num_episodes)
    prob_random = 0.05

    jvm.start()
    loader = Loader(classname="weka.core.converters.ArffLoader")
    #----------------------------------------------------------
    data1 = loader.load_file(arff1) #data/data_agent1_wPub.arff")
    data2 = loader.load_file(arff2) #data/data_agent2_wPub.arff")
    #cc1, cc2 = loader.load_file(cc1file), loader.load_file(cc2file)
    data1.class_is_last()
    data2.class_is_last()
    #cc1.class_is_last()
    #cc2.class_is_last()
    
    nA1 = int(max(act1)) + 1 #len(act)
    nA2 = int(max(act2)) + 1
    #----------------------------------------------------------
    
    cls1 = Classifier(classname=classifier_name)
    cls2 = Classifier(classname=classifier_name)
    
    #cls1 = Classifier(classname=classifier_name, options=options_NN)
    #cls2 = Classifier(classname=classifier_name, options=options_NN)
    
    cls1.build_classifier(data1)
    cls2.build_classifier(data2)
    
    fn = 'GN_IL_HAT_CCorC_Noise'+str(action_noise)+'_'+str(hat_decay)+'_alpha'+str(alpha)+'_cThresh'+str(CONF_THRESHOLD)+'_'+str(trial)
    outf, outstat = open(fn+'.raw','w'), open(fn+'.stat','w')
    trace = open('trace_'+str(trial)+'.raw', 'w')
    print("Trial: "+str(trial))
    
    lA1 = ql.LearningAgent(nF1, nA1, alpha, 0.999)
    lA2 = ql.LearningAgent(nF2, nA2, alpha, 0.999)
    cc1map, cc2map = load_coord_confidence( cc1file ), load_coord_confidence( cc2file )
    prob_HAT = 1.0
    #novel1, novel2 = {}, {}
    for episode in range(num_episodes):
        cnt_neither,cnt_CHAT, cnt_CC_same, cnt_CC_diff = 0,0,0,0
        #s1,s2 = gm.initState()
        #o1,o2 = gm.getPrivateObservations(s1,s2), gm.getPrivateObservations(s2,s1)
        #o1,o2 = gm.getPrivateObservations(s1), gm.getPrivateObservations(s2)
        #o = gm.getPublicObservation(s1,s2)
        o1, o2 = gm.reset()
        obs1, obs2 = tuple(o1), o2
        x1,x2=list(obs1),list(obs2)
        x1.append('100')
        x2.append('100') #append bad class
        inst1, inst2 = Instance.create_instance( x1 ), Instance.create_instance( x2 )
        #cc_inst1, cc_inst2 = Instance.create_instance( x1 ), Instance.create_instance( x2 )
        inst1.dataset, inst2.dataset = data1, data2
        #cc_inst1.dataset, cc_inst2.dataset = cc1, cc2
        step, episode_reward = 0, 0
        while step < max_num_steps:
            gm.render()
            #obs1, obs2 = o1+o, o2+o
            obs1, obs2 = tuple(o1), o2
            x1,x2=list(obs1),list(obs2)
            x1.append('100')
            x2.append('100') #append bad class
            #inst1, inst2 = Instance.create_instance( x1 ), Instance.create_instance( x2 )  
            #inst1.dataset, inst2.dataset = data1, data2

            for i in range(len(x1)-1):
                inst1.set_value(i,x1[i])
            for i in range(len(x2)-1):
                inst2.set_value(i,x2[i])
                #cc_inst1.set_value(i,x1[i])
                #cc_inst2.set_value(i,x2[i])
            #pA1, pA2 = cls1.classify_instance(inst1), cls2.classify_instance(inst2)
            dist1, dist2 = list( cls1.distribution_for_instance(inst1) ), list( cls2.distribution_for_instance(inst2) )

            chat_iA1, chat_iA2 = dist1.index( max(dist1) ), dist2.index( max(dist2) )
            chat_pA1, chat_pA2 = dist1[ chat_iA1 ], dist2[ chat_iA2 ]
            #pA1, pA2 = numpy.random.choice(numpy.arange(0, len(act)), p=dist1 ), numpy.random.choice(numpy.arange(0, len(act)), p=dist2 )
            l_o1, l_o2 = obs1, obs2
            
            if l_o1 in cc1map:
                cc_iA1, cc_pA1 = chat_iA1, cc1map[l_o1]
            #elif l_o1 in novel1:
            #    cc_iA1, cc_pA1 = novel1[ l_o1 ]
            else:
                keys = matchKey( 1, l_o1, cc1map )
                if len(keys)>0:
                    cc_pA1 = max( [cc1map[ k ] for k in keys] )
                    for k in keys:
                        if cc1map[ k ] == cc_pA1:
                            match_key = k
                    x1 = list( match_key )
                    for i in range(len(x1)):
                        inst1.set_value(i,x1[i])
                    dist1 = list( cls1.distribution_for_instance(inst1) )
                    cc_iA1 = dist1.index( max(dist1) )
                    #print "A1--", cc1, keys, [cc1map[ k ] for k in keys]
                else:
                    cc_iA1, cc_pA1 = chat_iA1, -0.0 #No confidence, so recommend same as CHAT
                #print "A1",l_o1, cc1
                #cc1map[l_o1] = cc1
                #cc1map[l_o1] = 1.0 #How about learn to predict this as well?
                #novel1[ l_o1 ] = cc_iA1, cc_pA1
                #print "A1",l_o1
                
            if l_o2 in cc2map:
                cc_iA2, cc_pA2 = chat_iA2, cc2map[l_o2]
            #elif l_o2 in novel2:
            #    cc_iA2, cc_pA2 = novel2[ l_o2 ]
            else:
                keys = matchKey( 2, l_o2, cc2map )
                if len(keys)>0:
                    cc_pA2 = max( [cc2map[ k ] for k in keys] )
                    for k in keys:
                        if cc2map[ k ] == cc_pA2:
                            match_key = k
                    x2 = list( match_key )
                    for i in range(len(x2)):
                        inst2.set_value(i,x2[i])
                    dist2 = list( cls2.distribution_for_instance(inst2) )
                    cc_iA2 = dist2.index( max(dist2) )
                    #print "A2--", cc2, keys, [cc2map[ k ] for k in keys]
                else:
                    cc_iA2, cc_pA2 = chat_iA2, -0.0
                #cc2map[l_o2] = cc2
                #print "A2",l_o2, cc2
                #cc2map[l_o2] = 1.0
                #novel2[ l_o2 ] = cc_iA2, cc_pA2
                #print "A2",l_o2
            
            #cp1, cp2 = cc_cls1.classify_instance(cc_inst1), cc_cls2.classify_instance(cc_inst2)
            x, y = random.random(), random.random()
            if (x < prob_HAT): # and ((dist1[pA1] >= 0.7) or (cc1 >= 0.7)): # * cp1:
                a1,txt1 = combine_decisions1(chat_iA1, cc_iA1, chat_pA1, cc_pA1)
                if txt1=="CHAT":
                    cnt_CHAT += 1
                elif txt1=="CC":
                    if chat_iA1 == cc_iA1:
                        cnt_CC_same += 1
                    else:
                        cnt_CC_diff += 1
                else:
                    cnt_neither += 1
                    
                '''
                #if txt=="CHAT":
                #    print "A1 (CHAT but not CC):",l_o1,chat_pA1,cc_pA1,s1,s2,gm.getActionName(int(act1[chat_iA1])), gm.getActionName(int(act1[cc_iA1]))
                #el
                if txt=="CC":
                    print "A1 (not CHAT but CC):",l_o1,chat_pA1,cc_pA1,s1,s2,gm.getActionName(int(act1[chat_iA1])), gm.getActionName(int(act1[cc_iA1]))
                    if chat_iA1 != cc_iA1:
                        print "++"
                '''
                if a1 < 0:
                    a1 = lA1.get_action( obs1, prob_random )
                    txt1 = 'IQL'
                else:
                    a1 = int(act1[ a1 ])
                
                #cp1 = dist1[ pA1 ]
            else:
                cnt_neither += 1
                a1 = lA1.get_action( obs1, prob_random )
                txt1 = 'IQL'
                #cp1 = 1.0
            if (y < prob_HAT): # and ((dist2[pA2] >= 0.7) or (cc2 >= 0.7)): # * cp2:
                a2,txt2 = combine_decisions1(chat_iA2, cc_iA2, chat_pA2, cc_pA2)
                if txt2=="CHAT":
                    cnt_CHAT += 1
                elif txt2=="CC":
                    if chat_iA2 == cc_iA2:
                        cnt_CC_same += 1
                    else:
                        cnt_CC_diff += 1
                else:
                    cnt_neither += 1
                '''
                #if txt=="CHAT":
                #    print "A2 (CHAT but not CC):",l_o2,chat_pA2,cc_pA2,s1,s2,gm.getActionName(int(act2[chat_iA2])), gm.getActionName(int(act2[cc_iA2]))
                #el
                if txt=="CC":
                    print "A2 (not CHAT but CC):",l_o2,chat_pA2,cc_pA2,s1,s2,gm.getActionName(int(act2[chat_iA2])), gm.getActionName(int(act2[cc_iA2]))
                    if chat_iA2 != cc_iA2:
                        print "++"
                '''
                if a2 < 0:
                    a2 = lA2.get_action( obs2, prob_random )
                    txt2 = 'IQL'
                else:
                    a2 = int(act2[ a2 ])
                #cp2 = dist2[ pA2 ]
            else:
                a2 = lA2.get_action( obs2, prob_random )
                txt2 = 'IQL'
                cnt_neither += 1
                #cp2 = 1.0
            print("Epi", episode, "step", step, o1, o2, a1, txt1, a2, txt2, file=trace)
            step +=1
            na1, na2 = a1, a2
            #s1,s2 = gm.transition( s1,s2,na1,na2,action_noise )
            
            (o1, o2), r, done, _ = gm.step( (na1,na2) )
            
            #o1,o2 = gm.getPrivateObservations(s1,s2), gm.getPrivateObservations(s2,s1)
            #o1,o2 = gm.getPrivateObservations(s1), gm.getPrivateObservations(s2)
            #o = gm.getPublicObservation(s1,s2)
            #assert cp1 > 0 and cp2 > 0
            #alpha1, alpha2 = alpha+(cp1-0.5)*0.004, alpha+(cp2-0.5)*0.004 #min(0.1, alpha/cc1map[l_o1]), min(0.1, alpha/cc2map[l_o2])

            #if gm.isGoal(s1,s2):

            episode_reward += r
            if done:
                #print episode, step
                lA1.q_update(alpha,l_o1,a1,+100,1,())
                lA2.q_update(alpha,l_o2,a2,+100,1,())
                break
            
            lA1.q_update(alpha,l_o1,a1,-1,1,o1)
            lA2.q_update(alpha,l_o2,a2,-1,1,o2)
            
        if (episode % SKIP_EPISODES == 0):
            outf.write(str(episode)+" "+str(episode_reward)+" "+str(step)+"\n")
        if (episode % (5*SKIP_EPISODES) == 0):
            outstat.write(str(episode)+" "+str(cnt_CHAT)+" "+str(cnt_CC_same)+" "+str(cnt_CC_diff)+" "+str(cnt_neither)+"\n")
        prob_HAT *= prob_HAT_decay
    #outf.write("\n")
    outf.close()
    outstat.close()
    trace.close()
    #print "(",novel1,novel2,") novel observations found."
    #lA1.printQVals(())
    #lA2.printQVals(())
    #lA1.saveQ(fn+'.qv1')
    #lA2.saveQ(fn+'.qv2')
    #lA.printQTable()
    #jvm.stop()

#do_thread((2000, 0.9999992, 0.001))


if __name__ == '__main__':
    pool_size = 8  # num processors
    pool = Pool(pool_size)

    #hdrates=[0.9999]
    hdrates=[0.999977] #, 0.99999] #[0.99990, 0.99995, 0.99999] slower decay has big initial dip
    #al = [0.002]
    al = [0.005] #[0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01] #, 0.1, 0.5] #alpha=0.05] #[0.01, 0.1, 0.5, 0.9] 0.1 has an initial dip but stable afterward; 0.5(&0.9) has no dip but unstable afterward, avgd 0.3 was also unstable afterward
    #for trial in range(num_trials):
    #    pool.apply_async(do_thread, (trial,))
    pool.map(do_thread, [(i,j,k) for i in range(num_trials) for j in hdrates for k in al] )
    print('Process Complete')
