# Input: See usage.
# Assumes an N (+1) column file: i x1 x2 ... xN, separated by ' '.
# Does: windows and smoothes each xk column.
# Output: A stream of N+1 columns: t y1 ... yN --windowed & smoothed
# Note: Every reported value is windowed over the same sized window, including t=0

import sys
import csv

nargs = len(sys.argv)
if nargs < 4:
    print("Usage: ", sys.argv[0], " <n-cols> <data-file> <window-size> [<filter-coefficient>]")
else:
    N, fName, windowSize = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    if nargs > 4:
        alpha = float(sys.argv[4])
    else:
        alpha = 1.0
    inf = open(fName, 'r')
    inp = csv.reader(inf, delimiter=' ')
    iBuffer = []  # window for i
    xiBuffer = []  # windows for xi
    for i in range(N):
        xiBuffer.append([])
    runningAvg = [0.0]*N
    firstPt = True
    for row in inp:
        if len(row) != N+1:
            print("wrong #cols:",len(row), row)
            continue
        else:
            if len(iBuffer) < windowSize:
                iBuffer.append(int(row[0]))
                for i in range(N):
                    xiBuffer[i].append(float(row[i+1]))
            else:
                s = str(iBuffer[0])+' '
                if firstPt:
                    AL = 1.0
                    firstPt = False
                else:
                    AL = alpha
                for i in range(N):
                    runningAvg[i] += AL * ((sum(xiBuffer[i]) / (1.0 * windowSize)) - runningAvg[i])
                    s += str(runningAvg[i])+' '
                    del xiBuffer[i][0]
                    xiBuffer[i].append(float(row[i+1]))
                print(s)
                del iBuffer[0]
                iBuffer.append(int(row[0]))
