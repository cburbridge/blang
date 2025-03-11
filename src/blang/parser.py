from .tokeniser import TokenSpec


class ParseError(Exception): ...


class Node:
    def __init__(self, typ, token=None, children=None, parent=None, tokens=None):
        self.token = token
        self.children = children or []
        self.parent = parent
        self.typ = typ

        self._tokens = tokens
        self._eaten = 0

    def eat(self, token, set_leaf=False, set_child=False):
        if len(self._tokens) < self._eaten + 1:
            raise ParseError
        if self._tokens[self._eaten] == token:
            if set_leaf:
                self.token = self._tokens[self._eaten]
            if set_child:
                self.children.append(Node("token", token=self._tokens[self._eaten]))
            self._eaten += 1
            return True
        raise ParseError

    def eat_child(self, ChildType):
        n = ChildType(self._tokens[self._eaten :])
        if n:
            self._eaten += n._eaten
            self.children.append(n)
            return True
        raise ParseError


def maybe(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ParseError:
            return False

    return wrapper


def parser(method):
    def wrapper(tokens):
        try:
            prototype_node = Node(method.__name__, tokens=tokens)
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


def OneOf(*args):
    def _OneOf(tokens):
        for Possible in args:
            node = Possible(tokens)
            if node:
                return node
        return None

    return _OneOf


@parser
def BaseType(node):
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
        TokenSpec.BOOL,
    ]:
        if maybe(node.eat)(basetype, set_leaf=True):
            return node
    return None


@parser
def RefType(node):
    node.eat(TokenSpec.LESS_THAN)

    if not maybe(node.eat_child)(RefType) and not maybe(node.eat_child)(BaseType):
        return None
    node.eat(TokenSpec.MORE_THAN)

    return node


@parser
def IdentifierRef(node):
    node.eat(TokenSpec.MORE_THAN)
    node.eat_child(Identifier)
    node.eat(TokenSpec.LESS_THAN)
    return node


@parser
def Array(node):
    node.eat_child(OneOf(BaseType, RefType))
    node.eat(TokenSpec.LSQBRACKET)
    node.eat(TokenSpec.INTEGER, set_child=True)
    node.eat(TokenSpec.RSQBRACKET)
    return node


Type = OneOf(Array, BaseType, RefType)  # order matters


@parser
def Identifier(node):
    node.eat(TokenSpec.IDENTIFIER, set_leaf=True)
    return node


@parser
def TypedIdentifier(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.COLON)
    node.eat_child(Type)
    return node


@parser
def Declaration(node):
    node.eat_child(TypedIdentifier)
    if maybe(node.eat)(TokenSpec.ASSIGN):
        node.eat_child(Expr)
    return node


@parser
def Assignment(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.ASSIGN)
    node.eat_child(Expr)
    return node


@parser
def Number(node):
    if maybe(node.eat)(TokenSpec.INTEGER, set_leaf=True) or node.eat(
        TokenSpec.FLOAT, set_leaf=True
    ):
        return node


@parser
def Return(node):
    node.eat(TokenSpec.RETURN)
    maybe(node.eat_child)(Expr)
    return node


@parser
def FuncCall(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.LPAREN)
    while True:
        if not maybe(node.eat_child)(Expr):
            break
        if not maybe(node.eat)(TokenSpec.COMMA):
            break
    node.eat(TokenSpec.RPAREN)
    return node


@parser
def ParameterList(node):
    node.eat(TokenSpec.LPAREN)
    while True:
        if not maybe(node.eat_child)(TypedIdentifier):
            break
        if not maybe(node.eat)(TokenSpec.COMMA):
            break
    node.eat(TokenSpec.RPAREN)

    return node


@parser
def FuncDef(node):
    node.eat(TokenSpec.DEF)
    node.eat_child(Identifier)
    node.eat_child(ParameterList)
    node.eat(TokenSpec.COLON)
    node.eat_child(Type)
    node.eat_child(Block)
    return node


@parser
def Block(node):
    node.eat(TokenSpec.LBRACE)
    while True:
        c = maybe(node.eat_child)(Statement)
        if not c:
            break
    node.eat(TokenSpec.RBRACE)
    return node


@parser
def CapturedExpression(node):
    node.eat(TokenSpec.LPAREN)
    node.eat_child(Expr)
    node.eat(TokenSpec.RPAREN)
    return node.children[0]


@parser
def DeRef(node):
    node.eat(TokenSpec.LESS_THAN)
    node.eat_child(Additive)
    node.eat(TokenSpec.MORE_THAN)
    return node


@parser
def Boolean(node):
    if maybe(node.eat)(TokenSpec.TRUE, set_leaf=True) or maybe(node.eat)(
        TokenSpec.FALSE, set_leaf=True
    ):
        return node
    return None


Factor = OneOf(
    FuncCall, Number, Boolean, Identifier, IdentifierRef, CapturedExpression, DeRef
)


@parser
def Additive(node):
    node.eat_child(Term)
    if maybe(node.eat)(TokenSpec.MINUS, set_leaf=True) or maybe(node.eat)(
        TokenSpec.PLUS, set_leaf=True
    ):
        node.eat_child(Additive)
        return node
    return node.children[0]


@parser
def Term(node):
    node.eat_child(Factor)
    if maybe(node.eat)(TokenSpec.ASTRISK, set_leaf=True) or maybe(node.eat)(
        TokenSpec.DIVIDE, set_leaf=True
    ):
        node.eat_child(Term)
        return node
    return node.children[0]


@parser
def Relational(node):
    node.eat_child(Additive)
    if (
        maybe(node.eat)(TokenSpec.MORE_THAN, set_leaf=True)
        or maybe(node.eat)(TokenSpec.LESS_THAN, set_leaf=True)
        or maybe(node.eat)(TokenSpec.MORE_THAN_EQ, set_leaf=True)
        or maybe(node.eat)(TokenSpec.LESS_THAN_EQ, set_leaf=True)
        or maybe(node.eat)(TokenSpec.EQUAL, set_leaf=True)
        or maybe(node.eat)(TokenSpec.NOT_EQ, set_leaf=True)
    ):
        node.eat_child(Additive)
        return node
    return node.children[0]


@parser
def LogicAnd(node):
    node.eat_child(Relational)
    if maybe(node.eat)(TokenSpec.AND):
        node.eat_child(Relational)
        return node
    return node.children[0]


@parser
def LogicOr(node):
    node.eat_child(LogicAnd)
    if maybe(node.eat)(TokenSpec.OR):
        node.eat_child(LogicAnd)
        return node
    return node.children[0]


Expr = LogicOr  # Additive
Statement = OneOf(FuncCall, FuncDef, Assignment, Declaration, Return)


@parser
def Module(node):
    while maybe(node.eat_child)(Statement):
        continue
    if node._eaten != len(node._tokens):
        print("unexpected token ", node._tokens[node._eaten])
        raise ParseError
    return node
