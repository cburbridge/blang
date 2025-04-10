import enum
import copy
from blang.exceptions import UnexpectedCharacterError
from dataclasses import dataclass
from typing import List

ALPHA_LOWER = "abcdefghijklmnopqrstuvwxyz"
ALPHA_UPPER = ALPHA_LOWER.upper()
NUMERAL = "0123456789"
PLUSMINUS = "+-"
DECIMAL = "."
UNDERSCORE = "_"
SPACE = " "


@dataclass
class Token:
    typ: None
    lineno: int
    colno: int
    text: str


class Repeat:
    def __init__(self, symbol, min=1, max=None):
        self.sym = symbol
        self.min = min
        self.max = max
        self.count = 0

    def hit(self):
        self.count += 1

    @property
    def is_full(self):
        return self.max and self.count == self.max


class Matcher:
    class FeedResult(enum.Enum):
        FAIL = enum.auto()
        CONTINUE = enum.auto()
        DONE = enum.auto()
        DONE_NOT_EATEN = enum.auto()

    def __init__(self, *args):
        # exp is a list of string | list
        # when its a list, its a or operation on all in the list
        # when its a string, its a or operation on the string entries
        self.exp = []
        self.active_exp = []  # working copy of self.exp

        on_repeat = False
        for a in args:
            if isinstance(a, Repeat):  # if its a repeat then join with next
                self.exp.extend([a.sym] * a.min)
                self.exp.append([a])
                on_repeat = True
            else:
                if on_repeat:
                    self.exp[-1].append(a)
                    on_repeat = False
                else:
                    self.exp.append(a)
        self.reset()

    @property
    def active(self):
        return len(self.active_exp) > 0 and self.eaten_count > 0

    @property
    def eaten_count(self):
        return len(self.content)

    def feed(self, c):
        if len(self.active_exp) < 1:
            self.is_failed = True
            return Matcher.FeedResult.FAIL
        next_mach = self.active_exp[0]
        if isinstance(next_mach, str):
            if c in next_mach:
                self.content += c
                self.active_exp.pop(0)
            else:
                self.is_failed = True
                return Matcher.FeedResult.FAIL
        if isinstance(next_mach, list):
            # or on the list, as have a repeater
            repeat = next_mach[0]
            mach = next_mach[1] if len(next_mach) > 1 else None
            if not repeat.is_full and c in repeat.sym:
                self.content += c
                repeat.hit()
                return Matcher.FeedResult.CONTINUE
            elif mach is None:
                self.active_exp.pop(0)
                return Matcher.FeedResult.DONE_NOT_EATEN
            elif c in mach:
                self.content += c
                self.active_exp.pop(0)
            else:
                self.is_failed = True
                return Matcher.FeedResult.FAIL

        if len(self.active_exp) > 0:
            return Matcher.FeedResult.CONTINUE
        else:
            return Matcher.FeedResult.DONE

    def reset(self):
        self.content = ""
        self.active_exp = copy.deepcopy(self.exp)  # coppies the repeats too, nice
        self.is_failed = False


class CommentMatcher:
    """Specialist matcher for catching comments."""

    def __init__(self, comment_char="#"):
        self.comment_char = comment_char
        self.reset()

    @property
    def active(self):
        return len(self.active_exp) > 0 and self.eaten_count > 0

    @property
    def eaten_count(self):
        return len(self.content)

    def feed(self, c):
        if self.is_failed:
            return Matcher.FeedResult.FAIL
        if self.in_comment:
            if c == "\n":
                return Matcher.FeedResult.DONE_NOT_EATEN
            self.content += c
            return Matcher.FeedResult.CONTINUE
        if c == self.comment_char:
            self.in_comment = True
            self.content += c
            return Matcher.FeedResult.CONTINUE
        self.is_failed = True
        return Matcher.FeedResult.FAIL

    def reset(self):
        self.content = ""
        self.in_comment = False
        self.is_failed = False


class StringMatcher:
    """Specialist matcher for catching strings."""

    def __init__(self, str_char='"'):
        self.str_char = str_char
        self.reset()

    @property
    def active(self):
        return len(self.active_exp) > 0 and self.eaten_count > 0

    @property
    def eaten_count(self):
        return len(self.content)

    def feed(self, c):
        if self.is_failed:
            return Matcher.FeedResult.FAIL
        if self.in_string:
            if c == self.str_char:
                return Matcher.FeedResult.DONE
            self.content += c
            return Matcher.FeedResult.CONTINUE
        if c == self.str_char:
            self.in_string = True
            self.content += c
            return Matcher.FeedResult.CONTINUE
        self.is_failed = True
        return Matcher.FeedResult.FAIL

    def reset(self):
        self.content = ""
        self.in_string = False
        self.is_failed = False
