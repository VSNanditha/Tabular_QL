path = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/'
# path = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/objecttransport/run9/'
filename_base_agent1 = 'confvalues_drop_a1_'
filename_base_agent2 = 'confvalues_drop_a2_'
file_range = 30

for i in range(file_range):
    agent1 = open(path + filename_base_agent1 + str(i) + '.raw', 'r')
    agent2 = open(path + filename_base_agent2 + str(i) + '.raw', 'r')
    output_file = open(path + 'conf_avg_files/conf_values_drop_' + str(i) + '.raw', 'w')
    for line1, line2 in zip(agent1, agent2):
        episode_a1 = int(line1[:line1.find(' ')])
        cc_confidence_a1 = float(line1[line1.find(' ') + 1:line1.find(' ', line1.find(' ') + 1)])
        chat_confidence_a1 = float(line1[line1.find(' ', line1.find(' ') + 1):line1.rfind(' ')])
        q_confidence_a1 = float(line1[line1.rfind(' ') + 1:])

        episode_a2 = int(line2[:line2.find(' ')])
        cc_confidence_a2 = float(line2[line2.find(' ') + 1:line2.find(' ', line2.find(' ') + 1)])
        chat_confidence_a2 = float(line2[line2.find(' ', line2.find(' ') + 1):line2.rfind(' ')])
        q_confidence_a2 = float(line2[line2.rfind(' ') + 1:])

        print(episode_a1, (cc_confidence_a1 + cc_confidence_a2) / 2,
              (chat_confidence_a1 + chat_confidence_a2) / 2,
              (q_confidence_a1 + q_confidence_a2) / 2, file=output_file)
    agent1.close()
    agent2.close()
