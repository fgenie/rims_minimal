import logging
import re
import signal
import numpy as np

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise Exception(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

# ==== https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py ====

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                logging.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                logging.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                logging.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        logging.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False

# these constants also used in OCW math below. do not detach this from here.
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# ==== for OCW (from minerva appendix) ==== 
INVALID_ANSWER = "[invalidanswer]"

def is_equiv_ocw(x1: str, x2: str)->bool: # to above, add numerical equivalence condition
    '''
    code took from Minerva original repository and adjusted for our use
    
    see OCWCourses::process_results
    https://github.com/wellecks/lm-evaluation-harness/blob/bec2172e72be4adc70e85957cc97a2fbe70c207b/lm_eval/tasks/ocw_courses.py#L153

    expects x1 and x2 to be string (latex)
    '''
    try:
        x1 = float(x1)
        x2 = float(x2)
        normalize_fn = normalize_numeric
        is_equiv = numeric_equality_ocw
    except ValueError as ve:
        if "=" in x1 or "=" in x2:
            normalize_fn = normalize_symbolic_equation
            is_equiv =  lambda x, y: x==y
            # answer_type = "equation" 
        else:
            normalize_fn = normalize_tex
            is_equiv = is_tex_equiv
            # answer_type = "expression"
    
    if INVALID_ANSWER in (x1, x2):
        return False
    
    

    
def numeric_equality_ocw(n1, n2, threshold=0.01):
    '''
    from appendix of the Minerva paper
    '''
    if n1 is None or n2 is None:
        return False
    if "None" in [n1, n2]:
        return False
    if np.isclose(n1,0) or np.isclose(n2,0) or np.isclose(n1-n2,0):
        return np.abs(n1-n2) < threshold * (n1+n2)/2
    else:
        return np.isclose(n1, n2)

def normalize_symbolic_equation(self, s: Optional[str]):
    if not isinstance(s, str):
        return INVALID_ANSWER
    if s.startswith("\\["):
        s = s[2:]
    if s.endswith("\\]"):
        s = s[:-2]
    s = s.replace("\\left(", "(")
    s = s.replace("\\right)", ")")
    s = s.replace("\\\\", "\\")
    if s.startswith("$") or s.endswith("$"):
        s = s.strip("$")
    try:
        maybe_expression = parse_latex(s)
        if not isinstance(maybe_expression, sympy.core.relational.Equality):
            # we have equation, not expression
            return INVALID_ANSWER
        else:
            return maybe_expression
    except:
        return INVALID_ANSWER




# ==== from Minerva code (https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/ocw_courses.py) ====
def process_results(self, doc, results, params={}):
    candidates = results[0]

    assert isinstance(params, dict)

    ref = doc['answer']

    try:
        float(ref)
        normalize_fn = self.normalize_numeric
        is_equiv = self.numeric_equality
        answer_type = "numeric"
    except ValueError:
        if "=" in ref:
            normalize_fn = self.normalize_symbolic_equation
            is_equiv = lambda x, y: x==y
            answer_type = "equation"
        else:
            normalize_fn = self.normalize_tex
            is_equiv = self.is_tex_equiv
            answer_type = "expression"

    correct_answer = normalize_fn(ref)

    if self.MAJORITY_VOTING not in params:
        unnormalized_answer = self.get_unnormalized_answer(candidates)

        model_answer = normalize_fn(unnormalized_answer)

        if unnormalized_answer == self.INVALID_ANSWER:
            acc = 0
        elif model_answer == self.INVALID_ANSWER:
            acc = 0
        elif is_equiv(model_answer, correct_answer):
            acc = 1
        else:
            acc = 0

        pass_rate = acc
    else:
        answers = [
            normalize_fn(self.get_unnormalized_answer(candidate))
            for candidate in candidates
            if self.get_unnormalized_answer(candidate) != self.INVALID_ANSWER
            and normalize_fn(self.get_unnormalized_answer(candidate)) != self.INVALID_ANSWER
        ]

        acc, pass_rate, votes = self.majority_vote(
            answers, correct_answer=correct_answer, is_equiv=is_equiv,
        )
        if votes:
            model_answer = votes[0][0]
        else:
            model_answer = self.INVALID_ANSWER

    results = {
        "acc": acc,
        "pass_rate": pass_rate,
        "metadata": {
            "selected_answer": model_answer, 
            "unprocessed_answers": candidates,
            "answer_type": answer_type,
        },
    }

    if self.MAJORITY_VOTING in params:
        results["metadata"]["votes"] = votes

    return results



def is_exp_equiv(self, x1: sympy.Basic, x2: sympy.Basic, time_limit=5) -> bool:
    """
    Determines whether two sympy expressions are equal.
    """
    try:
        with timeout(seconds=time_limit):
            try:
                diff = x1 - x2
            except (SympifyError, ValueError, TypeError) as e:
                print(
                    f"Couldn't subtract {x1} and {x2} with exception {e}"
                )
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except (SympifyError, ValueError, TypeError) as e:
                print(f"Failed to simplify {x1}-{x2} with {e}")
                return False
    except TimeoutError as e:
        print(f"Timed out comparing {x1} and {x2}")
        return False
    except Exception as e:
        print(f"failed on unrecognized exception {e}")
        return False

def is_tex_equiv(self, x1: str, x2: str, time_limit=5) -> bool:
    """
    Determines whether two (ideally normalized using `normalize_text`) TeX expressions are equal.

    Does so by first checking for string exact-match, then falls back on sympy-equivalence,
    following the (Lewkowycz et al. 2022) methodology.
    """
    if x1 == x2:
        # don't resort to sympy if we have full string match, post-normalization 
        return True

    parsed_x2 = self.parse_tex(x2)
    if not parsed_x2:
        # if our reference fails to parse into a Sympy object, 
        # we forgo parsing + checking our generated answer.
        return False
    return self.is_exp_equiv(self.parse_tex(x1), parsed_x2, time_limit=time_limit)



    # def numeric_equality(self, n1, n2, threshold=0.01):
    #     if n1 is None or n2 is None:
    #         return False
    #     if np.isclose(n1, 0) or np.isclose(n2, 0) or np.isclose(n1 - n2, 0):
    #         return np.abs(n1 - n2) < threshold * (n1 + n2) / 2
    #     else:
    #         return np.isclose(n1, n2)



def normalize_numeric(self, s):
    if s is None:
        return None
    for unit in [
        "eV",
        " \\mathrm{~kg} \\cdot \\mathrm{m} / \\mathrm{s}",
        " kg m/s",
        "kg*m/s",
        "kg",
        "m/s",
        "m / s",
        "m s^{-1}",
        "\\text{ m/s}",
        " \\mathrm{m/s}",
        " \\text{ m/s}",
        "g/mole",
        "g/mol",
        "\\mathrm{~g}",
        "\\mathrm{~g} / \\mathrm{mol}",
        "W",
        "erg/s",
        "years",
        "year",
        "cm",
    ]:
        s = s.replace(unit, "")
        s = s.strip()
    for maybe_unit in ["m", "s", "cm"]:
        s = s.replace("\\mathrm{" + maybe_unit + "}", "")
        s = s.replace("\\mathrm{~" + maybe_unit + "}", "")
        s = s.strip()
    s = s.strip("$")
    try:
        return float(eval(s))
    except:
        try:
            expr = parse_latex(s)
            if expr.is_number:
                return float(expr)
            return INVALID_ANSWER
        except:
            return INVALID_ANSWER
        

# this one from MajorityVotingMixin --> use this for get_concordant answer?
def majority_vote(
        self,
        sampled_answers: List[T],
        correct_answer: T,
        is_equiv : Callable[[T, T], bool] = lambda x, y: x==y,
        invalid_answer: T = None
):
    """
    Performs majority voting on a list of candidate answers. 
    Returns accuracy and pass rate checked against `correct_answer`.
    Supports arbitrary definitions of equivalence via `is_equiv` argument.
    
    Arguments:
        sampled_answers: List[T], list of sampled answers
        correct_answer: T, ground truth.
        is_equiv: Callable[[T, T], bool], a function that determines when two answers 
            should be treated as equivalent. Default is T-equivalence, i.e `lambda x y: x==y`.
        invalid_answer: T, answer that corresponds to a parsing failure from a sample. 
            If passed as arg, no votes for invalid answer should be counted, but it should
            count against pass_rate.
    Returns:
        acc: int, 0/1 for correct/incorrect
        pass_rate: float, proportion of `sampled_answers` equivalent to `correct_answer`
        votes: List[Tuple[T, int]], for each distinct answer, the amount of votes for that answer. 
            Sorted by descending amount of votes, so that `elected_answer==votes[0][0]`
    """
    if not sampled_answers:
        return 0, 0, []

    answer_votes = {}

    # we only count votes for successfully parsed answers, as we choose not
    # to allow a model to vote for [invalidanswer] as its response.
    # however, we do want to calculate pass_rate as a function of 
    # total K = *num. sampled answers*.
    if invalid_answer:
        valid_sampled_answers = [answer for answer in sampled_answers if answer != invalid_answer]
    else:
        valid_sampled_answers = sampled_answers

    for answer in valid_sampled_answers:
        if answer in answer_votes: 
            answer_votes[answer] += 1
        else:
            counted = False
            for ref in answer_votes:
                if is_equiv(answer, ref) and not counted:
                    answer_votes[ref] += 1
                    counted=True
            if not counted: 
                answer_votes[answer] = 1

    votes = list(sorted(answer_votes.items(), key=lambda x: -x[1]))

    elected_answer = votes[0][0]

    if is_equiv(correct_answer, elected_answer):
        acc = 1
        pass_rate = votes[0][1] / len(sampled_answers)
    else:
        acc = 0
        pass_rate = 0
        for candidate, num_votes in answer_votes.items():
            if is_equiv(correct_answer, candidate):
                pass_rate = num_votes / len(sampled_answers)
                break

    return acc, pass_rate, votes
