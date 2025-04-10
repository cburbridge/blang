from blang.parser import (
    TokenSpec,
    Block,
    Assignment,
    Additive,
    Factor,
    Type,
    ParameterList,
    FuncDef,
    TypedIdentifier,
    Return,
    Declaration,
    Identifier,
    ArrayItem,
    Range,
    Squelch,
    print_tree,
)


def test_block():
    s = """
    {
      fisher : u32 
      fisher = fun(45.98+19, 12,other(99))
      print(fisher)
      return 43
    }
    """

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t = Block(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_assign():
    s = "fisher = fun(45.98+19, 12,other(99))"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t = Assignment(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_assign_ref():
    s = "<fisher> = 10"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t = Assignment(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_term():
    s = "45 + 98-12 * (75.2 - 12)"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    t = Additive(tokens)
    print_tree(t)
    assert t._eaten == len(tokens)


def test_number_factor():
    s = "(654.978)"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    t = Factor(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)
    assert t.token.typ == TokenSpec.FLOAT
    assert float(t.token.text) == 654.978


def test_types():
    s = "u32"
    tokens = list(TokenSpec.tokenise(s))
    t = Type(tokens)
    assert t._eaten == 1
    # assert isinstance(t, BaseType)
    assert t.token.typ == TokenSpec.U32
    # print_tree(t)
    # assert False


def test_base_type():
    s = "u32"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type(tokens)
    print_tree(t)
    assert t._eaten == 1
    # assert isinstance(t, BaseType)
    # assert False


def test_ref_type():
    s = "<<u32>>"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type(tokens)
    print_tree(t)
    assert t._eaten == 5
    # assert isinstance(t, RefType)
    # assert False


def test_paramlist():
    s = "(fish:f64, face:<i8>) {}"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = ParameterList(tokens)
    assert t
    assert t._eaten == 11
    print(t.children)
    print_tree(t)
    # assert False


def test_func():
    #
    s = """ def MyFun(fish:f64, face:<i8>):u8{
    
       a: u8 = 9
       b: <u8> = >a<
       c: u8 = a;
       <b> = 9
       return c
    }
    """
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = FuncDef(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)


def test_typed_ident():
    s = "fish: f64"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = TypedIdentifier(tokens)
    assert t
    assert t._eaten == 3
    print(t.children)
    # assert isinstance(t, TypedIdentifier)
    print_tree(t)
    # assert False


def test_typed_ident_no_type():
    s = "fish"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = TypedIdentifier(tokens)
    assert not t


def test_identifier():
    s = "fish"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Identifier(tokens)
    assert t
    # assert t._eaten == 3
    print(t)
    print(t.children)
    print_tree(t)
    # assert False


def test_decl():
    s = "fisher :u32 = 99"
    # s = "99"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t = Declaration(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_return():
    s = "return 9"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Return(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)


def test_types2():
    s = "<<f64>>"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False


def test_array():
    s = "s:u8[10]"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Declaration(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False


def test_array_assign():
    s = "q=s[10]"
    tokens = list(TokenSpec.tokenise(s))
    t = Assignment(tokens)
    assert t
    print_tree(t)
    assert t._eaten == len(tokens)


def test_array_assign_lval():
    s = "s[10]=99"
    tokens = list(TokenSpec.tokenise(s))
    t = Assignment(tokens)
    assert t
    print_tree(t)
    assert t._eaten == len(tokens)


def test_array_item():
    s = "s[10]"
    tokens = list(TokenSpec.tokenise(s))
    t = ArrayItem(tokens)
    assert t
    assert t._eaten == len(tokens)
    print_tree(t)


def test_range():
    s = "1..100:5"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Range(tokens)
    assert t
    assert t._eaten == len(tokens)
    print_tree(t)


def test_range_no_step():
    s = "1..100"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Range(tokens)
    assert t
    assert t._eaten == len(tokens)
    print_tree(t)


def test_squelch():
    s = "u8|fish|"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Squelch(tokens)
    assert t
    assert t._eaten == len(tokens)
    print_tree(t)
