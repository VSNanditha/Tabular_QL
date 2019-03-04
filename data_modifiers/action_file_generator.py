path = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/blockdudes/run1/'
# path = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/"

window_size = 100
plot_filename_bases = []
file_range = 30

# Append all file_name prefixes to be plotted together
plot_filename_bases.append('action_choice_')

for i in range(len(plot_filename_bases)):
    for j in range(file_range):
        output_file = open(path + plot_filename_bases[i] + 'updated_' + str(j) + '.raw', 'w')
        input_file = open(path + plot_filename_bases[i] + str(j) + '.raw', 'r')
        for line in input_file:
            episode = int(line[:line.find(' ')])
            if (episode == 0) or ((episode + 1) % window_size == 0):
                print(line, file=output_file, end='')
        output_file.close()
