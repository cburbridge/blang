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
    s = "ref ref u32"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type(tokens)
    print_tree(t)
    assert t._eaten == 3
    # assert isinstance(t, RefType)
    # assert False


def test_paramlist():
    s = "(fish:f64, face:ref i8) {}"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = ParameterList(tokens)
    assert t
    assert t._eaten == 10
    print(t.children)
    print_tree(t)
    # assert False


def test_func():
    s = """
    def MyFun(fish:f64, face:ref i8):u8 {
    
       a: u8 = 9
       b: u8 = 10
       c: u8 = a +b
    
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
    s = "ref ref f64"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False
