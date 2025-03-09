from .tokeniser import TokenSpec


class ParseError(Exception): ...


class Node:
    def __init__(self, token=None, children=None, parent=None, tokens=None):
        self.token = token
        self.children = children or []
        self.parent = parent

        self._tokens = tokens
        self._eaten = 0

    def eat(self, token, set_leaf=False):
        if len(self._tokens) < self._eaten + 1:
            raise ParseError
        if self._tokens[self._eaten] == token:
            if set_leaf:
                self.token = self._tokens[self._eaten]
            self._eaten += 1
            return True
        raise ParseError

    def eat_child(self, node):
        n = node.create(self._tokens[self._eaten :])
        if n:
            self._eaten += n._eaten
            self.children.append(n)
            return True
        raise ParseError

    @property
    def typ(self):
        return self.__class__


def maybe(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ParseError:
            return False

    return wrapper


def parser(method):
    @classmethod
    def wrapper(cls, tokens):
        try:
            prototype_node = cls(tokens=tokens)
            created_node = method(prototype_node)
            if created_node:
                # if the created node is not the prototype, maybe as we just want a child,
                # then we need to stil consider the tokens eaten by the prototype as eatend by
                # the chosed node
                created_node._eaten = prototype_node._eaten
            return created_node
        except ParseError:
            return None

    return wrapper


"""
Statement       -> Assignment | FuncDef | FuncCall |
                   Declaration | ReturnStatement |
                   IfStatement | WhileLoop | ForLoop
Block           -> '{' Statement* '}'
Assignment      -> IDENTIFIER '=' Expr
Declaration     -> TypedIdentifier ('=' Expr)?
ReturnStatement -> 'return' Expr

Expr            -> LogicOr   # skip to Additive

LogicOr         -> LogicAnd ('or' LogicAnd)*      
LogicAnd        -> Equality ('and' Equality)*      
Equality        -> Relational (('==' | '!=') Relational)*  
Relational      -> Additive (('>' | '<' | '>=' | '<=') Additive)*

Additive        -> Term (('+' | '-') Term)*    
Term            -> Factor (('*' | '/') Factor)*
Factor          -> NUMBER | IDENTIFIER | FuncCall | '(' Expr ')' 

BaseType        -> 'u8' | 'u16' | 'u32' | 'u64' | 'u128' | 'i8' | 'i16' | 'i32' | 'i64' | 'i128' | 'f32' | 'f64'
Type            -> BaseType ('ref'*)
Identifier      -> IDENTIFIER
TypedIdentifier -> IDENTIFIER ':' Type

FuncDef         -> 'def' IDENTIFIER '(' TypedIdentifier (',' TypedIdentifier)* ')':' Type '{' Statement* '}'
FuncCall        -> IDENTIFIER '(' (Expr (',' Expr)*)? ')'

IfStatement     -> 'if' '(' Expr ')' Block ('else' Block)?
WhileLoop       -> 'while' '(' Expr ')' Block
ForLoop         -> 'for' '(' Assignment? ';' Expr? ';' Assignment? ')' Block


"""


class BaseType(Node):
    @parser
    def create(node):
        for basetype in [
            TokenSpec.U8,
            TokenSpec.U16,
            TokenSpec.U32,
            TokenSpec.U64,
            TokenSpec.U128,
            TokenSpec.I8,
            TokenSpec.I16,
            TokenSpec.I32,
            TokenSpec.I64,
            TokenSpec.I128,
            TokenSpec.F32,
            TokenSpec.F64,
        ]:
            if maybe(node.eat)(basetype, set_leaf=True):
                return node
        return None


class RefType(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.REF)

        if maybe(node.eat_child)(RefType):
            return node
        if maybe(node.eat_child)(BaseType):
            return node

        return None


def OneOf(*args):
    class _one_of:
        @classmethod
        def create(cls, tokens):
            for Possible in args:
                node = Possible.create(tokens)
                if node:
                    return node
            return None

    return _one_of


Type = OneOf(BaseType, RefType)  # order matters


class Identifier(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.IDENTIFIER, set_leaf=True)
        return node


class TypedIdentifier(Node):
    @parser
    def create(node):
        node.eat_child(Identifier)
        node.eat(TokenSpec.COLON)
        node.eat_child(Type)
        return node


class Declaration(Node):
    @parser
    def create(node):
        node.eat_child(TypedIdentifier)
        if maybe(node.eat)(TokenSpec.ASSIGN):
            node.eat_child(Expr)
        return node


class Assignment(Node):
    @parser
    def create(node):
        node.eat_child(Identifier)
        node.eat(TokenSpec.ASSIGN)
        node.eat_child(Expr)
        return node


class Number(Node):
    @parser
    def create(node):
        if maybe(node.eat)(TokenSpec.INTEGER, set_leaf=True) or node.eat(
            TokenSpec.FLOAT, set_leaf=True
        ):
            return node


class Return(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.RETURN)
        maybe(node.eat_child)(Expr)
        return node


class FuncCall(Node):
    @parser
    def create(node):
        node.eat_child(Identifier)
        node.eat(TokenSpec.LPAREN)
        while True:
            if not maybe(node.eat_child)(Expr):
                break
            if not maybe(node.eat)(TokenSpec.COMMA):
                break
        node.eat(TokenSpec.RPAREN)
        return node


class ParameterList(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.LPAREN)
        while True:
            if not maybe(node.eat_child)(TypedIdentifier):
                break
            if not maybe(node.eat)(TokenSpec.COMMA):
                break
        node.eat(TokenSpec.RPAREN)

        return node


class FuncDef(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.DEF)
        node.eat_child(Identifier)
        node.eat_child(ParameterList)
        node.eat(TokenSpec.COLON)
        node.eat_child(Type)
        node.eat_child(Block)
        return node


class Block(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.LBRACE)

        while True:
            c = maybe(node.eat_child)(Statement)
            if not c:
                break
        node.eat(TokenSpec.RBRACE)
        return node


class CapturedExpression(Node):
    @parser
    def create(node):
        node.eat(TokenSpec.LPAREN)
        node.eat_child(Expr)
        node.eat(TokenSpec.RPAREN)
        return node.children[0]


Factor = OneOf(FuncCall, Number, Identifier, CapturedExpression)


class Additive(Node):
    @parser
    def create(node):
        node.eat_child(Term)
        if maybe(node.eat)(TokenSpec.MINUS, set_leaf=True) or maybe(node.eat)(
            TokenSpec.PLUS, set_leaf=True
        ):
            node.eat_child(Additive)
            return node
        return node.children[0]


class Term(Node):
    @parser
    def create(node):
        node.eat_child(Factor)
        if maybe(node.eat)(TokenSpec.ASTRISK, set_leaf=True) or maybe(node.eat)(
            TokenSpec.DIVIDE, set_leaf=True
        ):
            node.eat_child(Term)
            return node
        return node.children[0]


Statement = OneOf(FuncCall, FuncDef, Assignment, Declaration, Return)

Expr = Additive


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
    t = Block.create(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_assign():
    s = "fisher = fun(45.98+19, 12,other(99))"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t = Assignment.create(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_term():
    s = "45 + 98-12 * (75.2 - 12)"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    t = Additive.create(tokens)
    print_tree(t)
    assert t._eaten == len(tokens)


def test_number_factor():
    s = "(654.978)"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    t = Factor.create(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)
    assert t.token.typ == TokenSpec.FLOAT
    assert float(t.token.text) == 654.978


def test_types():
    s = "u32"
    tokens = list(TokenSpec.tokenise(s))
    t = Type.create(tokens)
    assert t._eaten == 1
    assert isinstance(t, BaseType)
    assert t.token.typ == TokenSpec.U32
    # print_tree(t)
    # assert False


def test_base_type():
    s = "u32"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type.create(tokens)
    print_tree(t)
    assert t._eaten == 1
    assert isinstance(t, BaseType)
    # assert False


def test_ref_type():
    s = "ref ref u32"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type.create(tokens)
    print_tree(t)
    assert t._eaten == 3
    assert isinstance(t, RefType)
    # assert False


def test_paramlist():
    s = "(fish:f64, face:ref i8) {}"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = ParameterList.create(tokens)
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
    t = FuncDef.create(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False


def test_typed_ident():
    s = "fish: f64"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = TypedIdentifier.create(tokens)
    assert t
    assert t._eaten == 3
    print(t.children)
    assert isinstance(t, TypedIdentifier)
    print_tree(t)
    # assert False


def test_typed_ident_no_type():
    s = "fish"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = TypedIdentifier.create(tokens)
    assert not t


def test_identifier():
    s = "fish"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Identifier.create(tokens)
    assert t
    # assert t._eaten == 3
    print(t)
    print(t.children)
    print_tree(t)
    # assert False


def print_tree(node, indent=1):
    print(
        " " * (indent - 1),
        "-",
        node.typ,
        "  =",
        node.token.text if node.token else "XXXX",
    )
    for c in node.children:
        print_tree(c, indent + 3)
    # assert False


def test_decl():
    s = "fisher :u32 = 99"
    # s = "99"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t = Declaration.create(tokens)
    print_tree(t)
    # assert False
    assert t._eaten == len(tokens)


def test_return():
    s = "return 9"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Return.create(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)


def test_types2():
    s = "ref ref f64"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t = Type.create(tokens)
    assert t
    assert t._eaten == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False
