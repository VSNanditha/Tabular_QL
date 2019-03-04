# path = '/Users/nandithavittanala/virenv/tabularQL/action_selectors/'
# path = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/"
path = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run9/moving_obstacles/"

window_size = 100
plot_filename_bases = []
file_range = 10
gamma = 0.99

# Append all file_name prefixes to be plotted together
# plot_filename_bases.append('GN_IL_HAT_CCorC_Noise0.1_0.999977_alpha0.005_cThresh0.7_')
# plot_filename_bases.append('GN_IL_CHAT_Noise0.1_0.999977_alpha0.005_cThresh0.7_')
# plot_filename_bases.append('episode_rewards_drop_')
plot_filename_bases.append('episode_rewards_tabQL_')

for i in range(len(plot_filename_bases)):
    for j in range(file_range):
        output_file = open(path + 'temp/' + plot_filename_bases[i] + 'returns_' + str(j) + '.raw', 'w')
        input_file = open(path + plot_filename_bases[i] + str(j) + '.raw', 'r')
        for line in input_file:
            episode = int(line[:line.find(' ')])
            if episode % window_size == 0:
                # reward = float(line[line.find(' ') + 1:line.rfind(' ')])
                # steps = int(line[line.rfind(' ')+1:len(line)])
                reward = float(line[line.find(' ') + 1:line.find('.') + 2])
                steps = int(line[line.find('.') + 3:len(line)])
                episode_return = 0
                for k in range(1, steps + 1):
                    if reward == -200 and steps == 200:
                        episode_return += (-1) * (gamma ** k)
                    else:
                        if k == steps:
                            episode_return += 100 * (gamma ** k)
                        else:
                            episode_return += (-1) * (gamma ** k)
                print(episode, reward, file=output_file)
        output_file.close()
