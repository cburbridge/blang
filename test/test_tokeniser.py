import pytest
from blang.tokeniser import TokenSpec
import blang.exceptions


def test_tokeniser_valid():
    test_string = """
    def def_this_thing_def
       _stinks
    """
    expect = [
        TokenSpec.NEWLINE,
        TokenSpec.WHITESPACE,
        TokenSpec.DEF,
        TokenSpec.IDENTIFIER,
        TokenSpec.NEWLINE,
        TokenSpec.WHITESPACE,
        TokenSpec.IDENTIFIER,
        TokenSpec.NEWLINE,
        TokenSpec.WHITESPACE,
    ]
    actual = [t.typ for t in TokenSpec.tokenise(test_string)]
    assert actual == expect


def test_tokeniser_invalid():
    teststr = "something\nsomething~"
    with pytest.raises(blang.exceptions.UnexpectedCharacterError) as exceptioninfo:
        _ = list(TokenSpec.tokenise(teststr))
    assert exceptioninfo.value.line == 2
    assert exceptioninfo.value.c == "~"
    assert exceptioninfo.value.col == 10


def test_tokeniser_trick():
    teststr = "def a_def()"
    expect = [TokenSpec.DEF, TokenSpec.IDENTIFIER, TokenSpec.LPAREN, TokenSpec.RPAREN]
    actual = [t.typ for t in TokenSpec.tokenise(teststr)]
    assert actual == expect


def test_tokeniser_int():
    test = """123"""
    expect = [TokenSpec.INTEGER]
    actual = [t.typ for t in TokenSpec.tokenise(test)]
    assert actual == expect


def test_tokeniser_neg_int():
    test = """-123"""
    expect = [TokenSpec.MINUS, TokenSpec.INTEGER]
    actual = [t.typ for t in TokenSpec.tokenise(test)]
    assert actual == expect


def test_tokeniser_int_doubleneg():
    test = """--123"""
    expect = [TokenSpec.MINUS, TokenSpec.MINUS, TokenSpec.INTEGER]
    actual = [t.typ for t in TokenSpec.tokenise(test)]
    print(actual)
    assert actual == expect


def test_tokeniser_float():
    test = """1.5"""
    expect = [TokenSpec.FLOAT]
    actual = [t.typ for t in TokenSpec.tokenise(test)]
    print(actual)
    assert actual == expect
