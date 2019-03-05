# Input: Accepts a single file_name prefix, with multiple .stat files, plus other parameters
#       under CHANGE
# Do: Windows, then does average & standard-deviation over each column in .stat files, then
#    band_plots each column.
# Output: Creates a <prefix>plot.data file, in the directory where .stat files
#        are located. If not needed, this plot.data files should be deleted manually.

from curve_plotters.band_plot import *

# ======================================CHANGE======================================================
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run10/moving/"
dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run9/moving_obstacles/"
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/objecttransport/run9/conf_avg_files/"
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/action_selectors/"
fname_ID_range = range(10)
winsize = 20  # 200
alpha = 1.0
num_plot_points = 100
max_x = -1  # if not used

# plot_fname_base = 'action_choice_updated_'
# plot_fname_base = 'action_choice_drop_'
# plot_fname_base = 'conf_values_drop_'
# plot_fname_base = 'GN_IL_CHAT_Noise0.1_0.999977_alpha0.005_cThresh0.7_'
# plot_fname_base = 'GN_IL_HAT_CCorC_Noise0.1_0.999977_alpha0.005_cThresh0.7_'
# plot_fname_base = 'episode_rewards_drop_'
plot_fname_base = 'episode_rewards_tabQL_'


n_cols = 2  # y-cols in input .stat file (excluding x column)
plot_cols = [0, 1]  # column-indices to output
# plot_cols = [2, 3, 4]
# file_col_names = ['Episode Rewards CoordChat']
# file_col_names_map = {3:'CC = CHAT', 4:'CC != CHAT', 2:'Q'}
file_col_names_map = {0:'tabql_rewards', 1:'tabql_steps'}
# file_col_names = ['Avg CP (CC) vals', 'Avg CP (CHAT) vals', 'Avg CQ vals']
# file_col_names = ['rewards', 'steps']
line_types = [1, 3, 4, 8, 9, 10, 12, 13]
# ==================================================================================================

d, outf = create_plot_data(fname_ID_range, dir_string, plot_fname_base, winsize, alpha, ".raw", n_cols)
max_dsize = len(d)
data_key_skip = max(d.keys()) / max_dsize
max_num_keys = max_dsize if max_x < 0 else max_x / data_key_skip
e = max(1, int(max_num_keys / num_plot_points))
f = open('script', 'w')
# s = "set term postscript enhanced landscape color \"Arial\" 28\n" \
#     "set output \"BD_stats.eps\"\n" \
#     "set xlabel \'Training Episodes\'\nset ylabel \'Return\'\nset xtics 0,500000,3000000\n" \
#     "set key at 2500000,200\nset style fill transparent solid 0.3 noborder\n" \
#     "set xrange[0:3000000]\n"
s = "set key font \'Arial, 20\'\n" \
    "set xlabel \'Training Episodes\' font \'Arial, 15\'\nset ylabel \'Return\'font \'Arial, 15\'\n" \
    "set xtics 0,20000,100000 font \'Arial, 15\'\n" \
    "set ytics font \'Arial, 15\'\n" \
    "set key at 85000,40\nset style fill transparent solid 0.3 noborder\n" \
    "set xrange[0:100000]\n"
if max_x > 0:
    s += "set xrange[0:" + str(max_x) + "]\n"
s += "plot "
f.write(s)
s = ""
for i in plot_cols:  # range(n_cols):
    mu, sigma = (i + 1) * 2, (i + 1) * 2 + 1
    s += "'" + outf + "' every " + str(e) + " using 1:($" + str(mu) + "-$" + str(sigma) + "):($" + str(mu) + "+$" + str(
        sigma) \
         + ") with filledcurves notitle, '' every "
    s += str(e) + " using 1:" + str(mu) + " with lp lt " + str(line_types[plot_cols.index(i)]) + " pt " + str(i + 1) \
         + " ps 1.5 lw 2 title '" + file_col_names_map[i] + "',"
f.write(s)
f.close()
os.system('gnuplot -persist script')
