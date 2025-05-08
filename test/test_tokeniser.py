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
        TokenSpec.WHITESPACE,
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
    teststr = "def a_def() defdef"
    expect = [
        TokenSpec.DEF,
        TokenSpec.WHITESPACE,
        TokenSpec.IDENTIFIER,
        TokenSpec.LPAREN,
        TokenSpec.RPAREN,
        TokenSpec.WHITESPACE,
        TokenSpec.IDENTIFIER,
    ]
    actual = [t.typ for t in TokenSpec.tokenise(teststr)]
    assert actual == expect


def test_tokeniser_tricks():
    teststr = "u32u64"
    expect = [
        TokenSpec.IDENTIFIER,
    ]
    actual = [t.typ for t in TokenSpec.tokenise(teststr)]
    assert actual == expect


def test_tokeniser_tricks_typs():
    teststr = "u32 u64"
    expect = [
        TokenSpec.U32,
        TokenSpec.WHITESPACE,
        TokenSpec.U64,
    ]
    actual = [t.typ for t in TokenSpec.tokenise(teststr)]
    print([t.name for t in actual])
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
