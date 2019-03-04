# Input: Accepts multiple file_name prefixes, with multiple .raw files each, plus other parameters
#       under CHANGE
# Do: Windows, then does average & standard-deviation over .raw files for each prefix, then
#    band_plots each prefix.
# Output: Creates a <prefix>plot.data file for each prefix, in the directory where .raw files
#        are located. If not needed, these plot.data files should be deleted manually.


from curve_plotters.band_plot import *


# ======================================CHANGE======================================================
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/"
# dir_string = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/temp/';
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/objecttransport/run9/"
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/blockdudes/run1/temp/"
# dir_string = '/Users/nandithavittanala/virenv/tabularQL/action_selectors/'
# dir_string = '/Users/nandithavittanala/Downloads/'
dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run9/moving_obstacles/temp/"
fname_ID_range = range(10)
winsize = 20
alpha = 1.0  # leave it alone
num_plot_points = 100
max_x = -1  # -1 if not used
plot_fname_bases = []

# Append all file_name prefixes to be plotted together
# plot_fname_bases.append('episode_rewards_coordchat_returns_')
# plot_fname_bases.append('episode_rewards_chat_returns_')
plot_fname_bases.append('GN_IL_CHAT_Noise0.1_0.999977_alpha0.005_cThresh0.7_returns_')
plot_fname_bases.append('GN_IL_HAT_CCorC_Noise0.1_0.999977_alpha0.005_cThresh0.7_returns_')
plot_fname_bases.append('episode_rewards_drop_returns_')
plot_fname_bases.append('episode_rewards_tabQL_returns_')
# plot_fname_bases.append('Error_')

# plot_fname_bases.append('episode_rewards_coordchat_')
# plot_fname_bases.append('episode_rewards_chat_')
# plot_fname_bases.append('episode_rewards_drop_')
# plot_fname_bases.append('episode_rewards_tabQL_')

plot_names = ['CC', 'CHAT', 'DRoP', 'IQL']
# plot_names = ['CC', 'CHAT', 'IQL']
# plot_names = ['Tabular QL', 'CHAT', 'CC']
# plot_names = ['CHAT', 'CC', 'DRoP']
# plot_names = ['CHAT', 'CC']
# plot_names = ['Error']

line_types = [1, 3, 4, 7, 8, 9, 10, 12, 13]
# ==================================================================================================

out_fnames = []
max_dsize = 0
for f in plot_fname_bases:
    d, outf = create_plot_data(fname_ID_range, dir_string, f, winsize, alpha)
    out_fnames.append(outf)
    if len(d) > max_dsize:
        max_dsize = len(d)

e = max(1, int(max_dsize / num_plot_points))

# data_key_skip = max(d.keys()) / max_dsize
# max_num_keys = max_dsize if max_x < 0 else max_x / data_key_skip
# e = max(1, int(max_num_keys / num_plot_points))
os.system("rm x")
f = open('script', 'w')
# s = "set term postscript enhanced landscape color \"Arial\" 28\n" \
#     "set output \"episode_returns.eps\"\n" \
#     "set xlabel \'Training Episodes\'\nset ylabel \'Return\'\nset xtics 0,500000,3000000\n" \
#     "set key at 2500000,-50\nset style fill transparent solid 0.3 noborder\n" \
#     "set xrange[0:3000000]\n"
s = "set key font \'Arial, 20\'\n" \
    "set xlabel \'Training Episodes\' font \'Arial, 15\'\nset ylabel \'Return\'font \'Arial, 15\'\n" \
    "set font \'Arial, 15\'\n" \
    "set ytics font \'Arial, 15\'\n" \
    "set style fill transparent solid 0.3 noborder\n"
    # "set xrange[0:100000]\n"
if max_x > 0:
    s += "set xrange[0:" + str(max_x) + "]\n"
s += "plot "
f.write(s)
s=""
for i in range(len(out_fnames)):
    s += "'" + out_fnames[i] + "' every " + str(e) + " using 1:($2-$3):($2+$3) with filledcurves notitle, '' every "
    s += str(e) + " using 1:2 with lp lt " + str(line_types[i]) + " pt " + str(i + 1) + " ps 1.5 lw 2 title '" \
         + plot_names[i] + "',"
# print("s is",s)
f.write(s)
f.close()
os.system('gnuplot -persist script')
