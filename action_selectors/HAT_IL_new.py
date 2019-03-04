from multiprocessing import Pool
from two_agents import tabular_ql as ql
import random, math, numpy, gym

#==========================CHANGE==================================================================
#from domains import FurnitureMovers as gm
#from domains import BlockDudes as gm

gm = gym.make('GuideDog-v1')
num_trials = 8
action_noise = 0.1

#act1 = ['4', '2', '9', '8', '5', '1', '3', '7', '6', '0'] #FM
#act2 = ['4', '2', '9', '8', '5', '1', '3', '7', '6', '0'] #FM

#act1 = ['2', '3', '4', '1', '0'] #BD
#act2 = ['1', '3', '2', '4', '0'] #BD

#act1 = ['2', '4', '1', '0', '3'] #BDN
#act2 = ['3', '1', '4', '2', '0'] #BDN

#act1 = ['2', '3', '1', '4', '0'] #BDN-symm
#act2 = ['2', '3', '1', '4', '0'] #BDN-symm

act1 = ['0', '1', '2', '3', '4', '5', '6'] #GNdynamic
act2 = ['0', '1', '2', '3']                #GNdynamic

#arff1 = "arff/asymmetric_demo_bdn_noise_agent1_v1.arff" #"arff/symmetric_demo_fm_agent_v1.arff"
#arff2 = "arff/asymmetric_demo_bdn_noise_agent2_v1.arff" #"arff/symmetric_demo_fm_agent_v1.arff"

#arff1 = "arff/symmetric_demo_bdn_noise_agent_v1.arff" #BDN-symm
#arff2 = "arff/symmetric_demo_bdn_noise_agent_v1.arff" #BDN-symm

arff1 = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/demonstrations/final_arff/moving_obstacles/agent1.arff"
arff2 = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/demonstrations/final_arff/moving_obstacles/agent2.arff"

#arff1 = "arff/symmetric_demo_fm_noise_agent_v1.arff"
#arff2 = "arff/symmetric_demo_fm_noise_agent_v1.arff"

classifier_name = "weka.classifiers.trees.J48"
#classifier_name = "weka.classifiers.functions.MultilayerPerceptron"
#options_NN = ["-H", "20"] #NN

CONF_THRESHOLD = 0.7
nF1, nF2 = 11, 3
#==================================================================================================
    
#def noisy_jtaction( a1, a2 ): #takes desired a1, a2, and changes/not with noise 
#    x = random.random()
#    if (x < action_noise):
#        return random.randrange( len(act1) ), random.randrange( len(act2) )
#    else:
#        return a1, a2
        
#for trial in range(num_trials):
def do_thread( trip ): #trial, hat_decay, alpha) ):
    trial, hat_decay, alpha = trip[0], trip[1], trip[2]
    #print trial, hat_decay, alpha
    #return
    
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
    data1.class_is_last()
    data2.class_is_last()
    nA1 = int(max(act1)) + 1 #len(act)
    nA2 = int(max(act2)) + 1
    #----------------------------------------------------------
    cls1 = Classifier(classname=classifier_name)
    cls2 = Classifier(classname=classifier_name)

    #cls1 = Classifier(classname=classifier_name, options=options_NN)
    #cls2 = Classifier(classname=classifier_name, options=options_NN)
    
    cls1.build_classifier(data1)
    cls2.build_classifier(data2)

    fn = 'GN_IL_CHAT_Noise'+str(action_noise)+'_'+str(hat_decay)+'_alpha'+str(alpha)+'_cThresh'+str(CONF_THRESHOLD)+'_'+str(trial)
    outf = open(fn+'.raw','w')
    trace = open('chat_trace_' + str(trial) + '.raw', 'w')
    print("Trial: "+str(trial))
    
    lA1 = ql.LearningAgent(nF1, nA1, alpha, 0.999)
    lA2 = ql.LearningAgent(nF2, nA2, alpha, 0.999)
    prob_HAT = 1.0
    for episode in range(num_episodes):
        #s1,s2 = gm.stateInit()
        #o1,o2 = gm.getPrivateObservations(s1,s2), gm.getPrivateObservations(s2,s1)
        o1, o2 = gm.reset()
        #o = gm.getPublicObservation(s1,s2)
        obs1, obs2 = tuple(o1), o2
        x1,x2=list(obs1),list(obs2)
        x1.append('100')
        x2.append('100') #append bad class
        inst1, inst2 = Instance.create_instance( x1 ), Instance.create_instance( x2 )  
        inst1.dataset, inst2.dataset = data1, data2
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
                
            #pA1, pA2 = cls1.classify_instance(inst1), cls2.classify_instance(inst2)
            dist1, dist2 = list( cls1.distribution_for_instance(inst1) ), list( cls2.distribution_for_instance(inst2) )
            #pA1, pA2 = numpy.random.choice(numpy.arange(0, len(act)), p=dist1 ), numpy.random.choice(numpy.arange(0, len(act)), p=dist2 )
            pA1, pA2 = dist1.index( max(dist1) ), dist2.index( max(dist2) )
            
            l_o1, l_o2 = obs1, obs2
            x, y = random.random(), random.random()
            if (x < prob_HAT) and (dist1[pA1] >= CONF_THRESHOLD):
                a1 = int(act1[pA1])
                txt1 = 'CHAT'
            else:
                a1 = lA1.get_action( obs1, prob_random )
                txt1 = 'IQL'
            if (y < prob_HAT) and (dist2[pA2] >= CONF_THRESHOLD):
                a2 = int(act2[pA2])
                txt2 = 'CHAT'
            else:
                a2 = lA2.get_action( obs2, prob_random )
                txt2 = 'IQL'
            #print "Epi",episode, "step", step, o1, o2, a1, a2
            na1, na2 = a1, a2 #noisy_jtaction(a1, a2)
            step +=1


            #s1,s2 = gm.transition( s1,s2,na1,na2,action_noise )
            #o1,o2 = gm.getPrivateObservations(s1,s2), gm.getPrivateObservations(s2,s1)
            (o1, o2), r, done, _ = gm.step( (na1,na2) )
            print("Epi", episode, "step", step, o1, o2, a1, txt1, a2, txt2, file=trace)
            #o = gm.getPublicObservation(s1,s2)
            
            #alpha1, alpha2 = min(0.1, alpha/cc1map[l_o1]), min(0.1, alpha/cc2map[l_o2])
            #if gm.isGoal(s1,s2):
            episode_reward += r
            if done:
                #print episode, step
                lA1.q_update(alpha,l_o1,a1,+100,1,())
                lA2.q_update(alpha,l_o2,a2,+100,1,())
                break
            
            lA1.q_update(alpha,l_o1,a1,-1,1,o1)
            lA2.q_update(alpha,l_o2,a2,-1,1,o2)

        if episode % SKIP_EPISODES == 0:
            outf.write(str(episode)+" "+str(episode_reward)+" "+str(step)+"\n")
        prob_HAT *= prob_HAT_decay
    #outf.write("\n")
    outf.close()
    trace.close()
    #lA1.printQVals(())
    #lA2.printQVals(())
    #lA1.saveQ(fn+'.qv1')
    #lA2.saveQ(fn+'.qv2')
    #lA.printQTable()
    #jvm.stop()

#do_thread( (2000, 0.999977, 0.005) )


if __name__ == '__main__':
    pool_size = 8  # num processors
    pool = Pool(pool_size)

    hdrates=[0.999977] #0.99990, 0.999977, 0.99999] #[0.99990, 0.99995, 0.99999] slower decay has big initial dip
    al = [0.005] #[0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01] #[0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01] #, 0.1, 0.5] #alpha=0.05] #[0.01, 0.1, 0.5, 0.9] 0.1 has an initial dip but stable afterward; 0.5(&0.9) has no dip but unstable afterward, avgd 0.3 was also unstable afterward
    #for trial in range(num_trials):
    #    pool.apply_async(do_thread, (trial,))
    pool.map(do_thread, [(i,j,k) for i in range(num_trials) for j in hdrates for k in al] )

