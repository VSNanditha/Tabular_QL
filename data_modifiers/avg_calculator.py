# path = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/blockdudes/run1/'
path = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/"
# path = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run6/without_moving_objects/"

plot_filename_bases = []
file_range = 30

# Append all file_name prefixes to be plotted together
plot_filename_bases.append('episode_rewards_coordchat_')
# plot_filename_bases.append('episode_rewards_chat_')
# plot_filename_bases.append('episode_rewards_drop_')
# plot_filename_bases.append('episode_rewards_tabQL_')

for i in range(len(plot_filename_bases)):
    output_file = open(path + plot_filename_bases[i] + 'avg.raw', 'w')
    for j in range(file_range):
        input_file = open(path + plot_filename_bases[i] + str(j) + '.raw', 'r')
        step_op_file = open(path + 'cc_copy' + str(j) + '.raw', 'w')
        first_avg, last_avg = 0, 0
        for line in input_file:
            episode = int(line[:line.find(' ')])
            reward = float(line[line.find(' ') + 1:line.find('.') + 2])
            steps = int(line[line.find('.') + 3:len(line)])
            if episode % 100 == 0:
                print(episode, steps, file=step_op_file)
            if episode < 20000 or (episode >= 40000 and episode < 60000):
                # print(episode, steps, file=step_op_file)
                # print(steps)
                if episode < 20000:
                    first_avg += steps
                    print('first: ', episode, first_avg, steps)
                else:
                    last_avg += steps
                    print('last: ', episode, last_avg, steps)
        step_op_file.close()
        print(j, first_avg / 20000, last_avg / 20000, file=output_file)
    output_file.close()
