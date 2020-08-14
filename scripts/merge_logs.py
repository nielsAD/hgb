# -*- coding: utf-8 -*-

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

import sys
import glob
import getopt

def main():
    opt, arg = getopt.getopt(sys.argv[1:], 'hn:f:c:')
    opt = dict(opt)

    if ('-h' in opt):
        print('usage %s [-help] [-num_header=int] [-first_header=int] file1 .. fileN' % (sys.argv[0]))
        sys.exit()

    comment = (('-c' in opt) and opt['-c']) or None
    num_header = int(('-n' in opt) and opt['-n']) or 0
    first_header = int(('-f' in opt) and opt['-f']) or 0

    for i, f in enumerate((f for a in arg for f in glob.glob(a))):
        if comment is not None:
            print('#', f)
        with open(f) as o:
            for j, l in enumerate(o):
                if (i == 0 and j < first_header) or (i > 0 and j < num_header):
                    if comment is None:
                        continue
                    else:
                        sys.stdout.write(comment)
                sys.stdout.write(l)

if __name__ == '__main__':
    main()
