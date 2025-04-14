import copy
from .tokeniser import TokenSpec
import enum


class ParseError(Exception):
    def __init__(self, message, node):
        self.message = message
        self.node = node

    def __str__(self):
        return f"{self.message}"


class Node:
    def __init__(self, type, token=None, children=None, parent=None, tokens=None):
        self.token = token
        self.children = children or []
        self.parent = parent
        self.type = self.typ = type

        self._tokens = tokens
        self._eaten = 0

    def eat(self, token, set_leaf=False, set_child=False):
        if len(self._tokens) < self._eaten + 1:
            raise ParseError("unexpected end of input.", self)
        if self._tokens[self._eaten] == token:
            if set_leaf:
                self.token = self._tokens[self._eaten]
            if set_child:
                self.children.append(Node("token", token=self._tokens[self._eaten]))
            self._eaten += 1
            return True
        raise ParseError(f"expected {token}", self)

    def eat_child(self, ChildType):
        n = ChildType(self._tokens[self._eaten :])
        if n:
            self._eaten += n._eaten
            self.children.append(n)
            return True
        raise ParseError(f"Expected {ChildType}", self)

    @property
    def id(self):
        return f"{self._tokens[0].lineno}_{self._tokens[0].colno}"


def maybe(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ParseError as p:
            return False

    return wrapper


class NodeType(enum.StrEnum):
    UNKNOWN = enum.auto()
    BASE_TYPE = enum.auto()
    REF_TYPE = enum.auto()
    IDENTIFIER_REF = enum.auto()
    ARRAY = enum.auto()
    IDENTIFIER = enum.auto()
    TYPED_IDENTIFER = enum.auto()
    DECLARATION = enum.auto()
    ASSIGNMENT = enum.auto()
    INTEGER = enum.auto()
    FLOAT = enum.auto()
    RETURN = enum.auto()
    FUNC_CALL = enum.auto()
    PARAMETER_LIST = enum.auto()
    FUNC_DEF = enum.auto()
    FUNC_DECL = enum.auto()
    BLOCK = enum.auto()
    CAPTURED_EXPRESSION = enum.auto()
    DE_REF = enum.auto()
    ARRAY_ITEM = enum.auto()
    BOOLEAN = enum.auto()
    ADDITIVE = enum.auto()
    TERM = enum.auto()
    RELATIONAL = enum.auto()
    LOGIC_AND = enum.auto()
    LOGIC_OR = enum.auto()
    IF_STATEMENT = enum.auto()
    WHILE_LOOP = enum.auto()
    BREAK = enum.auto()
    RANGE = enum.auto()
    FOR_ARRAY_LOOP = enum.auto()
    FOR_RANGE_LOOP = enum.auto()
    MODULE = enum.auto()
    SQUELCH = enum.auto()
    STRING = enum.auto()
    CHARACTER = enum.auto()


def parser(type=NodeType.UNKNOWN):
    def parser(method):
        def wrapper(tokens):
            try:
                prototype_node = Node(type, tokens=tokens)
                created_node = method(prototype_node)
                if created_node:
                    # if the created node is not the prototype, maybe as we just want a child,
                    # then we need to stil consider the tokens eaten by the prototype as eatend by
                    # the chosen node
                    created_node._eaten = prototype_node._eaten
                return created_node
            except ParseError as p:
                return None

        return wrapper

    return parser


def print_tree(node, indent=1):
    if not node:
        print(">>>???<<<")
    else:
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


@parser(NodeType.BASE_TYPE)
def BaseType(node):
    for basetype in [
        TokenSpec.U8,
        TokenSpec.U16,
        TokenSpec.U32,
        TokenSpec.U64,
        TokenSpec.I8,
        TokenSpec.I16,
        TokenSpec.I32,
        TokenSpec.I64,
        TokenSpec.F64,
        TokenSpec.BOOL,
    ]:
        if maybe(node.eat)(basetype, set_leaf=True):
            return node
    return None


@parser(NodeType.REF_TYPE)
def RefType(node):
    node.eat(TokenSpec.LESS_THAN)

    if not maybe(node.eat_child)(RefType) and not maybe(node.eat_child)(BaseType):
        return None
    node.eat(TokenSpec.MORE_THAN)

    return node


@parser(NodeType.IDENTIFIER_REF)
def IdentifierRef(node):
    node.eat(TokenSpec.MORE_THAN)
    node.eat_child(Identifier)
    node.eat(TokenSpec.LESS_THAN)
    return node


@parser(NodeType.CHARACTER)
def Character(node):
    node.eat(TokenSpec.CHARACTER, set_leaf=True)
    return node


@parser(NodeType.ARRAY)
def Array(node):
    node.eat_child(OneOf(BaseType, RefType))
    node.eat(TokenSpec.LSQBRACKET)
    node.eat(TokenSpec.INTEGER, set_child=True)
    node.eat(TokenSpec.RSQBRACKET)
    return node


Type = OneOf(Array, BaseType, RefType)  # order matters


@parser(NodeType.SQUELCH)
def Squelch(node):
    node.eat_child(Type)
    node.eat(TokenSpec.BAR)
    node.eat_child(Factor)
    node.eat(TokenSpec.BAR)
    return node


@parser(NodeType.IDENTIFIER)
def Identifier(node):
    node.eat(TokenSpec.IDENTIFIER, set_leaf=True)
    return node


@parser(NodeType.TYPED_IDENTIFER)
def TypedIdentifier(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.COLON)
    node.eat_child(Type)
    return node


@parser(NodeType.DECLARATION)
def Declaration(node):
    node.eat_child(TypedIdentifier)
    if maybe(node.eat)(TokenSpec.ASSIGN):
        node.eat_child(ExprOrString)
    maybe(node.eat)(TokenSpec.TERMINATOR)
    return node


@parser(NodeType.DE_REF)
def DeRef(node):
    node.eat(TokenSpec.LESS_THAN)
    node.eat_child(Additive)
    node.eat(TokenSpec.MORE_THAN)
    return node


@parser(NodeType.ARRAY_ITEM)
def ArrayItem(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.LSQBRACKET)
    node.eat_child(Additive)
    node.eat(TokenSpec.RSQBRACKET)
    return node


LVal = OneOf(DeRef, ArrayItem, Identifier)  # order matters here


@parser(NodeType.ASSIGNMENT)
def Assignment(node):
    node.eat_child(LVal)
    node.eat(TokenSpec.ASSIGN)
    node.eat_child(ExprOrString)
    maybe(node.eat)(TokenSpec.TERMINATOR)
    return node


@parser(NodeType.INTEGER)
def Integer(node):
    node.eat(TokenSpec.INTEGER, set_leaf=True)
    return node


@parser(NodeType.FLOAT)
def Float(node):
    node.eat(TokenSpec.FLOAT, set_leaf=True)
    return node


Number = OneOf(Integer, Float)


@parser(NodeType.STRING)
def String(node):
    node.eat(TokenSpec.STRING, set_leaf=True)
    return node


@parser(NodeType.RETURN)
def Return(node):
    node.eat(TokenSpec.RETURN)
    maybe(node.eat_child)(Expr)
    maybe(node.eat)(TokenSpec.TERMINATOR)
    return node


@parser(NodeType.FUNC_CALL)
def FuncCall(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.LPAREN)
    while True:
        if not maybe(node.eat_child)(Expr):
            break
        if not maybe(node.eat)(TokenSpec.COMMA):
            break
    node.eat(TokenSpec.RPAREN)
    maybe(node.eat)(TokenSpec.TERMINATOR)
    return node


@parser(NodeType.PARAMETER_LIST)
def ParameterList(node):
    node.eat(TokenSpec.LPAREN)
    while True:
        if not maybe(node.eat_child)(TypedIdentifier):
            break
        if not maybe(node.eat)(TokenSpec.COMMA):
            break
    node.eat(TokenSpec.RPAREN)

    return node


@parser(NodeType.FUNC_DEF)
def FuncDef(node):
    node.eat(TokenSpec.DEF)
    node.eat_child(Identifier)
    node.eat_child(ParameterList)
    node.eat(TokenSpec.COLON)
    node.eat_child(Type)
    node.eat_child(Block)
    return node


@parser(NodeType.FUNC_DECL)
def FuncDecl(node):
    node.eat(TokenSpec.EXTERN)
    node.eat(TokenSpec.DEF)
    node.eat_child(Identifier)
    node.eat_child(ParameterList)
    if maybe(node.eat)(TokenSpec.COLON):
        node.eat_child(Type)
    maybe(node.eat)(TokenSpec.TERMINATOR)
    return node


@parser(NodeType.BLOCK)
def Block(node):
    node.eat(TokenSpec.LBRACE)
    while True:
        c = maybe(node.eat_child)(Statement)
        if not c:
            break
    node.eat(TokenSpec.RBRACE)
    return node


@parser(NodeType.CAPTURED_EXPRESSION)
def CapturedExpression(node):
    node.eat(TokenSpec.LPAREN)
    node.eat_child(Expr)
    node.eat(TokenSpec.RPAREN)
    return node.children[0]


@parser(NodeType.BOOLEAN)
def Boolean(node):
    if maybe(node.eat)(TokenSpec.TRUE, set_leaf=True) or maybe(node.eat)(
        TokenSpec.FALSE, set_leaf=True
    ):
        return node
    return None


Factor = OneOf(
    FuncCall,
    Number,
    Character,
    Boolean,
    ArrayItem,  # oh no, order matters :-(
    Identifier,
    IdentifierRef,
    CapturedExpression,
    DeRef,
    Squelch,
)


@parser(NodeType.ADDITIVE)
def Additive(node):
    node.eat_child(Term)
    while maybe(node.eat)(TokenSpec.MINUS, set_leaf=True) or maybe(node.eat)(
        TokenSpec.PLUS, set_leaf=True
    ):
        node.eat_child(Term)
        left = copy.deepcopy(node)
        node.children = []
        node.children.append(left)

    if len(node.children) == 1:
        return node.children[0]
    return node


# @parser(NodeType.TERM)
def _Term(node):
    node.eat_child(Factor)
    if maybe(node.eat)(TokenSpec.ASTRISK, set_leaf=True) or maybe(node.eat)(
        TokenSpec.DIVIDE, set_leaf=True
    ):
        node.eat_child(Term)
        return node
    return node.children[0]


@parser(NodeType.TERM)
def Term(node):
    node.eat_child(Factor)
    while (
        maybe(node.eat)(TokenSpec.ASTRISK, set_leaf=True)
        or maybe(node.eat)(TokenSpec.DIVIDE, set_leaf=True)
        or maybe(node.eat)(TokenSpec.MODULO, set_leaf=True)
    ):
        node.eat_child(Factor)
        left = copy.deepcopy(node)
        node.children = []
        node.children.append(left)

    if len(node.children) == 1:
        return node.children[0]
    return node


@parser(NodeType.RELATIONAL)
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


@parser(NodeType.LOGIC_AND)
def LogicAnd(node):
    node.eat_child(Relational)
    if maybe(node.eat)(TokenSpec.AND):
        node.eat_child(Relational)
        return node
    return node.children[0]


@parser(NodeType.LOGIC_OR)
def LogicOr(node):
    node.eat_child(LogicAnd)
    if maybe(node.eat)(TokenSpec.OR):
        node.eat_child(LogicAnd)
        return node
    return node.children[0]


@parser(NodeType.IF_STATEMENT)
def IfStatement(node):
    node.eat(TokenSpec.IF)
    node.eat_child(Expr)
    node.eat_child(Block)
    if maybe(node.eat)(TokenSpec.ELSE):
        node.eat_child(Block)
    return node


@parser(NodeType.WHILE_LOOP)
def WhileLoop(node):
    node.eat(TokenSpec.WHILE)
    node.eat_child(Expr)
    node.eat_child(Block)
    return node


@parser(NodeType.BREAK)
def Break(node):
    node.eat(TokenSpec.BREAK)
    maybe(node.eat)(TokenSpec.TERMINATOR)
    return node


@parser(NodeType.FOR_ARRAY_LOOP)
def ForArrayLoop(node):
    node.eat(TokenSpec.FOR)
    node.eat(TokenSpec.IDENTIFIER, set_child=True)
    node.eat(TokenSpec.AS)
    node.eat(TokenSpec.IDENTIFIER, set_child=True)
    node.eat(TokenSpec.COMMA)
    node.eat(TokenSpec.IDENTIFIER, set_child=True)
    node.eat_child(Block)
    return node


@parser(NodeType.FOR_RANGE_LOOP)
def ForRangeLoop(node):
    node.eat(TokenSpec.FOR)
    node.eat_child(Range)
    node.eat(TokenSpec.AS)
    node.eat_child(TypedIdentifier)
    node.eat_child(Block)
    return node


@parser(NodeType.RANGE)
def Range(node):
    node.eat_child(Integer)
    node.eat(TokenSpec.DOTDOT)
    node.eat_child(Integer)
    if maybe(node.eat)(TokenSpec.COLON):
        node.eat_child(Integer)
    return node


ForLoop = OneOf(ForArrayLoop, ForRangeLoop)

Expr = LogicOr  # Additive

ExprOrString = OneOf(Expr, String)

Statement = OneOf(
    FuncCall,
    # FuncDef,
    IfStatement,
    WhileLoop,
    ForLoop,
    Assignment,
    Declaration,
    Return,
    Break,
    Block,
)

DecOrDef = OneOf(FuncDef, FuncDecl, Declaration)


@parser(NodeType.MODULE)
def Module(node):
    while maybe(node.eat_child)(DecOrDef):
        continue
    if node._eaten != len(node._tokens):
        print("oh bugger unexpected token ", node._tokens[node._eaten])
        raise ParseError(f"unexpected token {node._tokens[node._eaten]}", node)
    return node
