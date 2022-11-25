#!/usr/bin/env python
import sys
import csv
import rtamt

def monitor():
    # Load traces
    req_1 = [[0, 0.6], [1, 2]]
    gnt_1 = [[0, 0.1], [1, 2.9]]

    req_2 = [[3, 0], [7, 3.1]]
    gnt_2 = [[5.6, 1], [7.9, 6]]

    # # #
    #
    # Example - online robustness
    #
    # # #
    spec = rtamt.STLDenseTimeSpecification()
    spec.name = 'Example 1'
    spec.declare_var('req', 'float')
    spec.declare_var('gnt', 'float')
    spec.declare_var('out', 'float')
    spec.set_var_io_type('req', 'input')
    spec.set_var_io_type('gnt', 'output')
    spec.spec = 'out = ((req>=3) implies (eventually[1:2](gnt>=3)))'
    try:
        spec.parse()
        spec.pastify()
    except rtamt.STLParseException as err:
        print('STL Parse Exception: {}'.format(err))
        sys.exit()

    rob = spec.update(['req', req_1], ['gnt', gnt_1])
    print('Robustness online - step 1: {}'.format(rob))

    rob = spec.update(['req', req_2], ['gnt', gnt_2])
    print('Robustness online - step 2: {}'.format(rob))

    #rob = spec.update_final(['req', []], ['gnt', []])
    #print('Robustness online - step final: {}'.format(rob))

if __name__ == '__main__':
    # Process arguments

    monitor()
