for i in range(30):
    f = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/confvalues_drop_a1_' + str(
        i) + '.raw'
    temp = open(f, 'r+')
    ep = 2000
    s = ''
    for line in temp:
        line = line.replace('(', '', 1)
        line = line.replace(')', '', 1)
        line = line.replace(', ', ' ', 2)
        line = str(ep) + ' ' + line
        s += line
        ep += 2000
    temp.close()
    new_file = '/Users/nandithavittanala/virenv/tabularQL/two_agents/rundata/guidedog/run8/with_moving_objects/confvalues_drop_a1_' + str(
        i) + '.raw'
    new_file = open(new_file, 'w')
    new_file.write(s)
    new_file.close()
