from utils.math_util import is_equiv_ocw

failure = "x_{0} \\cos (\\omega t)+$ $\\dot{x}_{0} \\sin (\\omega t) / \\omega"
print(is_equiv_ocw(failure, failure))
print(is_equiv_ocw(failure, failure, approach_w_symexp=True))