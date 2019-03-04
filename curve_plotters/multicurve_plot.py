# Input: A series of file names ending in '.raw', and having a common prefix
# fname_prefix. Set these in CHANGE section.
# Assumes 2 columns in each file: i xi, separated by ' '.
# Does: window each time series, plot as separate curves (for visualization only).
# Output: No Output; meant to aid parameter selection.

import os

# -----------------------CHANGE-----------------------------------------------------------------
directory_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/"

fname_prefix = 'cc_copy'
# fname_prefix = 'episode_rewards_chat  '
# fname_prefix = 'episode_rewards_tabQL'

winsize = 5  # window size
alpha = 1.0  # leave it alone

num_plot_points = 1000  # how many points to plot per curve?
e = 1  # every how many points? (will skip e-1 points between consecutive plot points)
maxX = -1  # max of X-axis; -1 if unused
winsum_string = "python /Users/nandithavittanala/virenv/tabularQL/curve_plotters/winsum.py "

# ----------------------------------------------------------------------------------------------

outf = open('script', 'w')
files = [i for i in os.listdir(directory_string) if os.path.isfile(os.path.join(directory_string,i)) and fname_prefix in i and '.raw' in i]
s = "plot "
if maxX>0:
    s += "[0:"+str(maxX)+"]"
for fid in range(len(files)):
    out_fname = "tmp_"+str(fid)
    windowing_string = winsum_string + ' 1 ' + directory_string + files[fid]+ " "+str(winsize) + " " + str(alpha) + " > "+out_fname
    print('windowing_string: ', windowing_string)
    os.system(windowing_string)
    s += "'"+out_fname+"' every "+str(e)+" using 1:2 with l lt "+str(fid)+" lw 1 title '"+files[fid]+"',"

outf.write(s)
outf.close()
os.system('gnuplot -persist script')

for fid in range(len(files)):
    os.system("rm tmp_"+str(fid))
os.system("rm script")
