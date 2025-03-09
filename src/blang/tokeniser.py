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

    def __eq__(self, o):
        if self.typ == o:
            return True
        return False


class TokenSpec(enum.Enum):
    DEF = Matcher(*"def")
    RETURN = Matcher(*"return")
    NEWLINE = Matcher("\n")
    IDENTIFIER = Matcher(
        ALPHA_LOWER + ALPHA_UPPER + UNDERSCORE,
        Repeat(ALPHA_LOWER + ALPHA_UPPER + NUMERAL + UNDERSCORE, min=0),
    )
    WHITESPACE = Matcher(Repeat(" \t", 1))
    LPAREN = Matcher("(")
    RPAREN = Matcher(")")
    LBRACE = Matcher("{")
    RBRACE = Matcher("}")
    COLON = Matcher(":")
    MINUS = Matcher("-")
    PLUS = Matcher("+")
    DIVIDE = Matcher("/")
    ASTRISK = Matcher("*")
    COMMA = Matcher(",")
    FLOAT = Matcher(Repeat(NUMERAL, min=0), PERIOD, Repeat(NUMERAL, min=1))
    INTEGER = Matcher(Repeat(NUMERAL, min=1))
    ASSIGN = Matcher("=")

    ####
    ## Types
    ##
    U8 = Matcher(*"u8")
    U16 = Matcher(*"u16")
    U32 = Matcher(*"u32")
    U64 = Matcher(*"u64")
    U128 = Matcher(*"u128")
    I8 = Matcher(*"i8")
    I16 = Matcher(*"i16")
    I32 = Matcher(*"i32")
    I64 = Matcher(*"i64")
    I128 = Matcher(*"i128")
    F32 = Matcher(*"f32")
    F64 = Matcher(*"f64")
    REF = Matcher(*"ref")
    ####

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

        # Append a special \0 to force that there is no active matcher
        # at the end of the string.
        input_text += "\0"
        fed = 0
        while True:
            c = input_text[p]
            p += 1
            fed += 1
            col_number += 1

            # Feed each active matcher with the character
            # and record any completed
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
                    if p == len(input_text):
                        break
                    raise UnexpectedCharacterError(c, line_number, col_number - 1)

                # Track the location in the input
                if token.typ == cls.NEWLINE:
                    line_number += 1
                    col_number = 1

                # Reset matchers and status
                for matcher in cls.all_matchers():
                    matcher.reset()
                active = set({t for t in cls})
                succeeded = []
                fed = 0

                if token.typ not in (TokenSpec.WHITESPACE, TokenSpec.NEWLINE):
                    yield token

            # Don't process the added \0
            if c == "\0":  # p == len(input_text) - 1:
                break


"""
        # If there are any active matchers left, that means there is some unconsumed data
        # It could be that there is an active matcher that is looking for repeat, but
        # could be closed if nothing else.


        for token in active:
            print(f"Condier {token.name}")
            token_matcher = token.value

            if token_matcher.feed("\0") == Matcher.FeedResult.DONE_NOT_EATEN:
                if token not in (TokenSpec.WHITESPACE, TokenSpec.NEWLINE):
                    # If the best of the succeeded token matchers was as good as this one
                    # then yield it instead
                    if len(succeeded) > 0:
                        choice = sorted(
                            succeeded, key=lambda t: t.value.eaten_count, reverse=True
                        )[0]
                        best_completed_token = Token(
                            typ=choice,
                            colno=col_number,
                            lineno=line_number,
                            text=choice.value.content,
                        )
                        if len(best_completed_token.text) == len(token_matcher.content):
                            yield best_completed_token
                            break

                    yield Token(
                        typ=token,
                        colno=col_number,
                        lineno=line_number,
                        text=token_matcher.content,
                    )
                    break
"""
