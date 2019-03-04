# Input: A series of file names ending in '..integer.raw'.
# Do: window each time series, find mean and sd, plot mean w confidence bands (for visualization only)
# Output: create a plot data file (x y ystderror) w a proper file name in the input directory

import os
import csv

# -----------------------CHANGE-----------------------------------------------------
# dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/run5/"
dir_string = "/Users/nandithavittanala/virenv/tabularQL/two_agents/domain_runners/"

fname_base_string = "episode_rewards_coordchat_"
# fname_base_string = 'QLambda_'

n_cols = 2  # x y1 .. yN; N=n_cols
fname_ID_range = range(1)  # number of files in each series
winsize = 100
alpha = 1.0
num_plot_points = 100
winsum_string = "python /Users/nandithavittanala/virenv/tabularQL/curve_plotters/winsum.py"
# ----------------------------------------------------------------------------------


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)


def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss


def stddev(data, ddof=1):
    """Calculates the population standard deviation when ddof=0
    otherwise computes the sample standard deviation."""
    n = len(data)
    if n < 2:
        #print "HERE", data
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


def create_plot_data(fname_ID_range, dir_string, fname_base_string, winsize, alpha, ext=".raw",n_cols=1):
    data_map = {}
    if ext==".raw":
        outfname = dir_string + fname_base_string + "plot.data"
        print("outfname = ",outfname)
    else:
        outfname = dir_string + fname_base_string + "plot"+ext+".data"
    outf = open(outfname, 'w')
                     
    for fid in fname_ID_range:
        windowing_string = winsum_string+' '+str(n_cols)+' '+dir_string + fname_base_string + str(fid) +ext+' '+str(winsize) + " " + str(alpha) + " > x"
        print("windowing_string = ",windowing_string)
        os.system(windowing_string)
        # t = input("Wait")
        inp = csv.reader(open('x','r'), delimiter=' ')
        # print('inp len: ', len(inp))
        # print('inp: ', inp)
        for row in inp:
            # print('row: ', row)
            x = int(row[0])
            # print('x: ', x)
            if x not in data_map:
                data_map[x] = []
                for i in range(n_cols):
                    data_map[x].append([])
            for i in range(n_cols):
                y = float(row[i+1])
                data_map[x][i].append( y )

    print('data_map', len(data_map.keys()), data_map,data_map.keys())
    for x in sorted(data_map.keys()):
        s = str(x) + ' ' 
        for i in range(n_cols):
            if (len(data_map[x][i]) >= 2):
                avg = mean(data_map[x][i])
                sd = stddev(data_map[x][i])
                s += str(avg) + ' ' + str(sd) + ' '
            else:
                # print("Too few points for Key=",x, "Column=",i, len(data_map[x][i]))
                s += '0 0 '
                # avg = mean(data_map[x][i])
                # s += str(avg)
        outf.write(s+'\n')

    outf.close()
    # os.system("rm x")
    return data_map, outfname


if __name__ == '__main__':
    data_map, outfname = create_plot_data(fname_ID_range, dir_string, fname_base_string, winsize, alpha)
    e = int( len(data_map) / num_plot_points )
    f = open('script','w')
    s = "set style fill transparent solid 0.2 noborder\n"
    f.write(s)
    s = "plot '"+outfname+"' every "+str(e)+" using 1:($2-$3):($2+$3) with filledcurves notitle, '' every "
    s+=str(e)+" using 1:2 with lp lt 1 pt 7 ps 1.5 lw 2 title '"+fname_base_string+"'"
    f.write(s)
    f.close()
    os.system('gnuplot -persist script')
os.system("rm script")
