from .tokeniser import TokenSpec


class ParseError(Exception): ...


class Node:
    def __init__(self, token=None, children=None, parent=None, tokens=None):
        self.token = token
        self.children = children or []
        self.parent = parent

        self._tokens = tokens
        self._eaten = 0

    def eat(self, token):
        if len(self._tokens) < self._eaten + 1:
            raise ParseError
        if self._tokens[self._eaten] == token:
            self._eaten += 1
            return True
        else:
            print(f"{__name__} Not eat => {self._tokens[self._eaten].typ}")
        raise ParseError

    def eat_child(self, node):
        n, c = node.create(self._tokens[self._eaten :])
        print(f"Child eaten {c=}")
        if n:
            self._eaten += c
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
    def wrapper(cls, *args, **kwargs):
        try:
            return method(cls, *args, **kwargs)
        except ParseError:
            return (None, 0)

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
    def create(cls, tokens):
        node = cls(token=tokens[0], tokens=tokens)
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
            if maybe(node.eat)(basetype):
                return node, node._eaten
        return None, 0


class RefType(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat(TokenSpec.REF)

        if maybe(node.eat_child)(RefType):
            return node, node._eaten
        if maybe(node.eat_child)(BaseType):
            return node, node._eaten

        return None, 0


def OneOf(*args):
    class _one_of:
        @parser
        def create(cls, tokens):
            for Possible in args:
                node, eaten = Possible.create(tokens)
                if node:
                    return node, eaten
            return None, 0

    return _one_of


Type = OneOf(BaseType, RefType)  # order matters


def test_types2():
    s = "ref ref f64"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = Type.create(tokens)
    assert t
    # assert c == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False


class Identifier(Node):
    @parser
    def create(cls, tokens):
        node = cls(token=tokens[0], tokens=tokens)
        node.eat(TokenSpec.IDENTIFIER)
        return node, node._eaten


class TypedIdentifier(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat_child(Identifier)

        # if not maybe(node.eat)(TokenSpec.COLON):
        # return node.children[0], node._eaten  # just an idenetifier
        node.eat(TokenSpec.COLON)
        node.eat_child(Type)
        return node, node._eaten


class Declaration(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat_child(TypedIdentifier)
        if maybe(node.eat)(TokenSpec.ASSIGN):
            node.eat_child(Expr)
        return node, node._eaten


def test_decl():
    s = "fisher :u32 = 99"
    # s = "99"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t, c = Declaration.create(tokens)
    print_tree(t)
    # assert False
    assert c == len(tokens)


class Assignment(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat_child(Identifier)
        node.eat(TokenSpec.ASSIGN)
        node.eat_child(Expr)
        return node, node._eaten


class Number(Node):
    @parser
    def create(cls, tokens):
        node = cls(token=tokens[0], tokens=tokens)
        if maybe(node.eat)(TokenSpec.INTEGER) or node.eat(TokenSpec.FLOAT):
            return node, node._eaten


class Return(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat(TokenSpec.RETURN)
        maybe(node.eat_child)(Expr)
        return node, node._eaten


def test_return():
    s = "return 9"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = Return.create(tokens)
    assert t
    assert c == len(tokens)
    print(t.children)
    print_tree(t)


class FuncCall(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat_child(Identifier)
        node.eat(TokenSpec.LPAREN)
        while True:
            if not maybe(node.eat_child)(Expr):
                break
            if not maybe(node.eat)(TokenSpec.COMMA):
                break
        node.eat(TokenSpec.RPAREN)
        return node, node._eaten


class ParameterList(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat(TokenSpec.LPAREN)
        while True:
            if not maybe(node.eat_child)(TypedIdentifier):
                break
            if not maybe(node.eat)(TokenSpec.COMMA):
                break
        node.eat(TokenSpec.RPAREN)

        return node, node._eaten


class FuncDef(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat(TokenSpec.DEF)
        node.eat_child(Identifier)
        node.eat_child(ParameterList)
        node.eat(TokenSpec.COLON)
        node.eat_child(Type)
        node.eat_child(Block)
        return node, node._eaten


class Block(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat(TokenSpec.LBRACE)

        while True:
            c = maybe(node.eat_child)(Statement)
            if not c:
                break
        node.eat(TokenSpec.RBRACE)
        return node, node._eaten


class CapturedExpression(Node):
    @parser
    def create(cls, tokens):
        node = cls(tokens=tokens)
        node.eat(TokenSpec.LPAREN)
        node.eat_child(Expr)
        node.eat(TokenSpec.RPAREN)
        return node.children[0], node._eaten  # noteice potential eat count isssue


Factor = OneOf(FuncCall, Number, Identifier, CapturedExpression)


class Term(Node):
    @parser
    def create(cls, tokens):
        total_eaten = 0
        node, eaten = Factor.create(tokens)
        if node:
            total_eaten += eaten
            while total_eaten < len(tokens):
                if (
                    tokens[total_eaten] == TokenSpec.ASTRISK
                    or tokens[total_eaten] == TokenSpec.DIVIDE
                ):
                    inner_eaten = 1
                    factor2, eaten = Factor.create(tokens[total_eaten + inner_eaten :])
                    if factor2:
                        node = Node(tokens[total_eaten], children=[node, factor2])
                        total_eaten += inner_eaten + eaten
                    else:
                        break
                else:
                    break
            return node, total_eaten
        return None, 0


class Additive(Node):
    @parser
    def create(cls, tokens):
        total_eaten = 0
        node, eaten = Term.create(tokens)
        if node:
            total_eaten += eaten
            while total_eaten < len(tokens):
                if (
                    tokens[total_eaten] == TokenSpec.PLUS
                    or tokens[total_eaten] == TokenSpec.MINUS
                ):
                    inner_eaten = 1
                    term2, eaten = Term.create(tokens[total_eaten + inner_eaten :])
                    if term2:
                        node = Node(tokens[total_eaten], children=[node, term2])
                        total_eaten += inner_eaten + eaten
                    else:
                        break
                else:
                    break
            return node, total_eaten
        return None, 0


# class Statement(Node):
#    @parser
#    def create(cls, tokens):
#        for Possible in (
#            FuncCall,
#            FuncDef,
#            Assignment,
#            Declaration,
#            Return,
#        ):  # todo: order he matters
#            node, eaten = Possible.create(tokens)
#            if node:
#                return node, eaten
#        return None, 0

Statement = OneOf(FuncCall, FuncDef, Assignment, Declaration, Return)

# class Expr(Node):
# @classmethod
# def create(cls, tokens):
# node, eaten = Number.create(tokens)
# if node:
# return node, eaten
# return None, 0
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
    t, c = Block.create(tokens)
    print_tree(t)
    # assert False
    assert c == len(tokens)


def test_assign():
    s = "fisher = fun(45.98+19, 12,other(99))"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    print()
    t, c = Assignment.create(tokens)
    print_tree(t)
    # assert False
    assert c == len(tokens)


def test_term():
    s = "45 + 98 * (75.2 - 12)"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    t, c = Additive.create(tokens)
    print_tree(t)
    # assert False
    assert c == len(tokens)


def test_number_factor():
    s = "(654.978)"

    tokens = list(TokenSpec.tokenise(s))
    for t in tokens:
        print(t.typ.name)
    t, c = Factor.create(tokens)
    print_tree(t)
    # assert False
    assert c == len(tokens)
    assert t.token.typ == TokenSpec.FLOAT
    assert float(t.token.text) == 654.978


def test_types():
    s = "u32"
    tokens = list(TokenSpec.tokenise(s))
    t, c = Type.create(tokens)
    assert c == 1
    assert isinstance(t, BaseType)
    assert t.token.typ == TokenSpec.U32
    # print_tree(t)
    # assert False


def test_base_type():
    s = "u32"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = Type.create(tokens)
    print_tree(t)
    assert c == 1
    assert isinstance(t, BaseType)
    # assert False


def test_ref_type():
    s = "ref ref u32"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = Type.create(tokens)
    print_tree(t)
    assert c == 3
    assert isinstance(t, RefType)
    # assert False


def test_paramlist():
    s = "(fish:f64, face:ref i8) {}"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = ParameterList.create(tokens)
    assert t
    assert c == 10
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
    t, c = FuncDef.create(tokens)
    assert t
    assert c == len(tokens)
    print(t.children)
    print_tree(t)
    # assert False


def test_typed_ident():
    s = "fish: f64"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = TypedIdentifier.create(tokens)
    assert t
    assert c == 3
    print(t.children)
    assert isinstance(t, TypedIdentifier)
    print_tree(t)
    # assert False


def test_typed_ident_no_type():
    s = "fish"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = TypedIdentifier.create(tokens)
    assert not t


def test_identifier():
    s = "fish"
    tokens = list(TokenSpec.tokenise(s))
    print([t.typ.name for t in tokens])
    t, c = Identifier.create(tokens)
    assert t
    # assert c == 3
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
