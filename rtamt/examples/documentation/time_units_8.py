import sys
import rtamt

def monitor():
    spec = rtamt.STLDiscreteTimeSpecification()
    spec.name = 'Bounded-response Request-Grant'

    spec.declare_var('req', 'float')
    spec.declare_var('gnt', 'float')
    spec.declare_var('out', 'float')
    
    spec.unit = 'ms'
    spec.set_sampling_period(1, 's', 0.1)

    spec.spec = 'out = (req>=3) implies (eventually[500s:1500s](gnt>=3))'

    try:
        spec.parse()
        spec.pastify()
        spec.update(0, [('req', 0.1), ('gnt', 0.3)])
        spec.update(500, [('req', 0.45), ('gnt', 0.12)])
        spec.update(1000, [('req', 0.78), ('gnt', 0.18)])
        nb_violations = spec.sampling_violation_counter
        print(nb_violations)
    except rtamt.STLParseException as err:
        print('STL Parse Exception: {}'.format(err))
        sys.exit()

if __name__ == '__main__':
    # Process arguments
    monitor()

