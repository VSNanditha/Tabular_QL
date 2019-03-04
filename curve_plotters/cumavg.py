# Input: See usage
# Assumes an N (+1) column file: i x1 x2 ... xN, separated by ' '.
# Does: cumulative avg of each column.
# Output: A stream of N+1 columns: t y1 ... yN --cumulative-averaged
# Note: Reported values are windowed over ever-expanding windows,
#      but starting from a specified minimum size.

import csv
import sys

nargs = len(sys.argv)
if nargs < 4:
    print("Usage: ", sys.argv[0], " <n-cols> <data file> <min. window size>")
else:
    N, fName, min_window = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    inf = open(fName, 'r')
    inp = csv.reader(inf, delimiter=' ')
    iBuffer = []  # window for i
    xiBuffer = []  # windows for xi
    for i in range(N):
        xiBuffer.append([])
    runningAvg = [0.0] * N
    for row in inp:
        if len(row) != N+1:
            # print "wrong #cols:",len(row), row
            continue
        else:
            cur_x = int(row[0])
            iBuffer.append(cur_x)
            for i in range(N):
                num = float(row[i+1])
                xiBuffer[i].append( num )
            if len(xiBuffer[i]) < min_window:
                continue
            s = str(iBuffer[0])+' '
            for i in range(N):
                runningAvg[i] = (sum(xiBuffer[i]) / (1.0 * len(xiBuffer[i])))
                s += str(runningAvg[i])+' '
            print(s)
            del iBuffer[0]
