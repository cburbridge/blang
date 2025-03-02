import enum
from blang.exceptions import UnexpectedCharacterError
from dataclasses import dataclass
from typing import List
from .matcher import Matcher, Repeat

ALPHA_LOWER = "abcdefghijklmnopqrstuvwxyz"
ALPHA_UPPER = ALPHA_LOWER.upper()
NUMERAL = "0123456789"
PLUSMINUS = "+-"
PERIOD = "."
UNDERSCORE = "_"
SPACE = " "
EQ = "="


@dataclass
class Token:
    typ: None
    lineno: int
    colno: int
    text: str


class TokenSpec(enum.Enum):
    DEF = Matcher(*"def ")
    NEWLINE = Matcher("\n")
    IDENTIFIER = Matcher(
        ALPHA_LOWER + ALPHA_UPPER + UNDERSCORE,
        Repeat(ALPHA_LOWER + ALPHA_UPPER + NUMERAL + UNDERSCORE, min=0),
    )
    WHITESPACE = Matcher(Repeat(" \t", 1))
    LPAREN = Matcher("(")
    RPAREN = Matcher(")")
    COLON = Matcher(":")
    MINUS = Matcher("-")
    PLUS = Matcher("+")
    DIVIDE = Matcher("/")
    MULTIPLY = Matcher("*")
    FLOAT = Matcher(Repeat(NUMERAL, min=0), PERIOD, Repeat(NUMERAL, min=1))
    INTEGER = Matcher(Repeat(NUMERAL, min=1))
    ASSIGN = Matcher("=")

    @classmethod
    def all_matchers(cls):
        return [t.value for t in cls]

    @classmethod
    def tokenise(cls, input_text) -> List[Token]:
        for matcher in cls.all_matchers():
            matcher.reset()
        p = 0
        line_number = 1
        col_number = 1

        active = set({t for t in cls})
        succeeded = []

        fed = 0
        while p < len(input_text):
            c = input_text[p]
            p += 1
            fed += 1
            col_number += 1

            for token in list(active):
                token_matcher = token.value
                match token_matcher.feed(c):
                    case Matcher.FeedResult.CONTINUE:
                        pass
                    case Matcher.FeedResult.FAIL:
                        active.remove(token)
                    case _:
                        succeeded.append(token)
                        active.remove(token)

            if len(active) == 0:
                if len(succeeded) > 0:
                    choice = sorted(
                        succeeded, key=lambda t: t.value.eaten_count, reverse=True
                    )[0]
                    token = Token(
                        typ=choice,
                        colno=col_number,
                        lineno=line_number,
                        text=choice.value.content,
                    )
                    ret = fed - len(token.text)
                    p -= ret
                    col_number -= ret
                else:
                    raise UnexpectedCharacterError(c, line_number, col_number - 1)

                if token.typ == cls.NEWLINE:
                    line_number += 1
                    col_number = 1

                for matcher in cls.all_matchers():
                    matcher.reset()
                active = set({t for t in cls})
                succeeded = []
                fed = 0

                yield token

        # If there are any active matchers left, that means there is some unconsumed data
        # It could be that there is an active matcher that is looking for repeat, but
        # could be closed if nothing else.

        for token in active:
            token_matcher = token.value

            if token_matcher.feed("\0") == Matcher.FeedResult.DONE_NOT_EATEN:
                yield Token(
                    typ=token,
                    colno=col_number,
                    lineno=line_number,
                    text=token_matcher.content,
                )
                break
