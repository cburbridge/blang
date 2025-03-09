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

    def eat(self, token, set_leaf=False):
        if len(self._tokens) < self._eaten + 1:
            raise ParseError
        if self._tokens[self._eaten] == token:
            if set_leaf:
                self.token = self._tokens[self._eaten]
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
    ]:
        if maybe(node.eat)(basetype, set_leaf=True):
            return node
    return None


@parser
def RefType(node):
    node.eat(TokenSpec.REF)

    if maybe(node.eat_child)(RefType):
        return node
    if maybe(node.eat_child)(BaseType):
        return node

    return None


Type = OneOf(BaseType, RefType)  # order matters


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


Factor = OneOf(FuncCall, Number, Identifier, CapturedExpression)


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


Statement = OneOf(FuncCall, FuncDef, Assignment, Declaration, Return)

Expr = Additive
