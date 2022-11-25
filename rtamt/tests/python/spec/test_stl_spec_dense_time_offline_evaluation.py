import unittest
import rtamt

class TestSTLBooleanAndTemporalOffline(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSTLBooleanAndTemporalOffline, self).__init__(*args, **kwargs)

    def test_and(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req and gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 1.3], [0.7, 3], [1.3, -1.2], [2.1, -2.2]]


        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "and")


    def test_or(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req or gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5], [0.7, 4], [1.3, 0.1], [2.1, 1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "or")

    def test_iff(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req iff gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 1.3 - 2.5], [0.7, 3 - 4], [1.3, -1.2 - 0.1], [2.1, -2.2 - 1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "iff")

    def test_xor(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req xor gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5-1.3], [0.7, 4-3], [1.3, 1.2 + 0.1], [2.1, 2.2 + 1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "xor")

    def test_implies(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req implies gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5], [0.7, 4], [1.3, -0.1], [2.1, 2.2]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "implies")


    def test_always(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = always(req)'

        spec.parse()

        op = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, -1.2], [2.1, 1.7]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "always")

    def test_eventually(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = eventually(req)'

        spec.parse()

        op = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 4], [1.3, 1.7], [2.1, 1.7]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "ev")


    def test_historically(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = historically(req)'

        spec.parse()

        op = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5], [1.3, -1.2], [2.1, -1.2]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "hist")

    def test_once(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = once(req)'

        spec.parse()

        op = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5], [0.7, 4], [2.1, 4]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "hist")


    def test_since(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req since gnt'

        spec.parse();
        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        self.assertListEqual(out, expected, "since")

    def test_until(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req until gnt'

        spec.parse();
        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 1.3], [0.7, 3], [1.3, -1.2], [2.1, -2.2]]
        self.assertListEqual(out, expected, "until")

    def test_once_0_1(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = once[0:1](req)'

        spec.parse()

        op = [[0, 4], [5, 2], [10, 5], [15, 3]]

        expected = [[0, 4], [6, 2], [10, 5], [15, 5]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "once[0,1]")

    def test_historically_0_1(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = historically[0:1](req)'

        spec.parse()

        op = [[0, 4], [5, 2], [10, 5], [15, 3]]

        expected = [[0, 4], [5, 2], [11, 5], [15, 3]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "hist[0,1]")

    def test_always_0_1(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = always[0:1](req)'

        spec.parse()

        op = [[0, 4], [5, 2], [10, 5], [15, 3]]

        expected = [[0, 4], [4, 2], [10, 5], [15, 5]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "alw[0,1]")

    def test_eventually_0_1(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = eventually[0:1](req)'

        spec.parse()

        op = [[0, 4], [5, 2], [10, 5], [15, 3]]

        expected = [[0, 4], [5, 2], [9, 5], [15, 5]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "ev[0,1]")

    def test_since_0_1(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req since[0:1] gnt'

        spec.parse();
        left = [[0, 2], [10, 2]]
        right = [[0, 4], [10, 4]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 2], [10, 2]]
        self.assertListEqual(out, expected, "since[0:1]")

    def test_until_0_1(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req until[0:1] gnt'

        spec.parse();
        left = [[0, 2], [10, 2]]
        right = [[0, 4], [10, 4]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 2], [10, 2]]
        self.assertListEqual(out, expected, "until[0:1]")


    def test_addition(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req + gnt'

        spec.parse();
        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 1.3 + 2.5], [0.7, 3 + 4], [1.3, 0.1 + -1.2], [2.1, -2.2 + 1.7]]
        self.assertListEqual(out, expected, "add")

    def test_subtraction(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req - gnt'

        spec.parse();
        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 1.3 - 2.5], [0.7, 3 - 4], [1.3, 0.1 - -1.2], [2.1, -2.2 - 1.7]]
        self.assertListEqual(out, expected, "sub")

    def test_multiplication(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req * gnt'

        spec.parse();
        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 1.3 * 2.5], [0.7, 3 * 4], [1.3, 0.1 * -1.2], [2.1, -2.2 * 1.7]]
        self.assertListEqual(out, expected, "mult")

    def test_division(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req / gnt'

        spec.parse();
        left = [[0, 1.3], [0.7, 3.], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4.], [1.3, -1.2], [2.1, 1.7]]

        out = spec.evaluate(['req', left], ['gnt', right])
        expected = [[0, 1.3 / 2.5], [0.7, 3. / 4.], [1.3, 0.1 / -1.2], [2.1, -2.2 / 1.7]]
        self.assertListEqual(out, expected, "div")

    def test_abs(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = abs(req)'

        spec.parse()

        op = [[1.3, 4], [3.7, -2.2], [9.4, -33]]

        expected = [[1.3, 4], [3.7, 2.2], [9.4, 33]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "abs")


    def test_not(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = not(req)'

        spec.parse()

        op = [[1.3, 4], [3.7, -2.2], [9.4, -33]]

        expected = [[1.3, -4], [3.7, 2.2], [9.4, 33]]
        computed = spec.evaluate(['req', op])

        self.assertListEqual(expected, computed, "not")

    def test_next(self):
        spec = rtamt.STLDenseTimeSpecification()
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = next req'

        spec.parse()

        self.assertRaises(rtamt.LTLNotImplementedException, spec.update, ['req', [0.0, 3.0]])

    def test_prev(self):
        spec = rtamt.STLDenseTimeSpecification()
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = prev req'

        spec.parse()

        self.assertRaises(rtamt.LTLNotImplementedException, spec.update, ['req', [0.0, 3.0]])

    def test_rise(self):
        spec = rtamt.STLDenseTimeSpecification()
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = rise(req)'

        spec.parse()

        self.assertRaises(rtamt.LTLNotImplementedException, spec.update, ['req', [0.0, 3.0]])

    def test_fall(self):
        spec = rtamt.STLDenseTimeSpecification()
        spec.declare_var('req', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = fall(req)'

        spec.parse()

        self.assertRaises(rtamt.LTLNotImplementedException, spec.update, ['req', [0.0, 3.0]])



    def test_predicate_leq(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req <= gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5 - 1.3], [0.7, 4 - 3], [1.3, -1.2-0.1], [2.1, 1.7+2.2]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "leq")

    def test_predicate_less(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req < gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5 - 1.3], [0.7, 4 - 3], [1.3, -1.2 - 0.1], [2.1, 1.7 + 2.2]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "less")

    def test_predicate_geq(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req >= gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 1.3-2.5], [0.7, 3-4], [1.3, 0.1+1.2], [2.1, -2.2-1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "geq")

    def test_predicate_greater(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req > gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 1.3 - 2.5], [0.7, 3 - 4], [1.3, 0.1 + 1.2], [2.1, -2.2 - 1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "greater")

    def test_predicate_eq(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req == gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 1.3 - 2.5], [0.7, 3 - 4], [1.3, -0.1 - 1.2], [2.1, -2.2 - 1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "eq")

    def test_predicate_neq(self):
        spec = rtamt.STLDenseTimeSpecification();
        spec.declare_var('req', 'float')
        spec.declare_var('gnt', 'float')
        spec.declare_var('out', 'float')
        spec.spec = 'out = req !== gnt'

        spec.parse();

        left = [[0, 1.3], [0.7, 3], [1.3, 0.1], [2.1, -2.2]]
        right = [[0, 2.5], [0.7, 4], [1.3, -1.2], [2.1, 1.7]]

        expected = [[0, 2.5-1.3], [0.7, 4-3], [1.3, 0.1 + 1.2], [2.1, 2.2+1.7]]

        computed = spec.evaluate(['req', left], ['gnt', right])

        self.assertListEqual(expected, computed, "neq")

if __name__ == '__main__':
    unittest.main()