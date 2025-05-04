import copy
from dataclasses import dataclass
import enum
from .tokeniser import TokenSpec


class ParseError(Exception):
    def __init__(self, message, node):
        self.message = message
        self.node = node

    def __str__(self):
        return f"{self.message}"


@dataclass
class NodeExtra:
    """Extra detail added to a node before compiling but after parsing."""

    precompiled: bool = False
    is_literal: bool = False
    data_type: str | None = None
    potential_size_mismatch: bool = False
    is_lvalue: bool = False
    is_rvalue: bool = False
    produces_value: bool = False


class Node:
    def __init__(self, type, token=None, children=None, parent=None, tokens=None):
        self.token = token
        self.children = children or []
        self.parent = parent
        self.type = self.typ = type

        self._tokens = tokens
        self._eaten = 0
        self.blang = ""
        self.extra = NodeExtra()

    def eat(
        self,
        token,
        set_leaf=False,
        set_child=False,
        ignore_space=True,
        ignore_linebreak=True,
    ):
        if len(self._tokens) < self._eaten + 1:
            raise ParseError("unexpected end of input.", self)

        # Eat any whitespace before the token
        original_eaten = self._eaten
        ignores = []
        if ignore_linebreak:
            ignores.append(TokenSpec.NEWLINE)
        if ignore_space:
            ignores.append(TokenSpec.WHITESPACE)
        while (
            ignores
            and len(self._tokens) - 1 > self._eaten
            and self._tokens[self._eaten].typ in ignores
        ):
            self._eaten += 1

        if self._tokens[self._eaten] == token:
            if set_leaf:
                self.token = self._tokens[self._eaten]
            if set_child:
                self.children.append(Node(token, token=self._tokens[self._eaten]))
            self._eaten += 1
            return True

        self._eaten = original_eaten  # restore it not to steal the whitespace
        raise ParseError(f"expected {token}", self)

    def eat_child(self, ChildType):
        n = ChildType(self._tokens[self._eaten :])
        if n:
            self._eaten += n._eaten
            self.children.append(n)
            return True
        raise ParseError(f"Expected {ChildType}", self)

    def peek_child(self, ChildType):
        n = ChildType(self._tokens[self._eaten :])
        return n

    @property
    def id(self):
        return f"{self._tokens[0].lineno}_{self._tokens[0].colno}"

    @property
    def indent_count(self):
        indent = 0
        first_none_white = 0
        while self._tokens[first_none_white].typ in (
            TokenSpec.WHITESPACE,
            TokenSpec.NEWLINE,
        ):
            first_none_white += 1

        # count white spaces spaces backwards
        ix = first_none_white - 1
        while self._tokens[ix].typ == TokenSpec.WHITESPACE:
            indent += len(self._tokens[ix].text)
            ix -= 1

        return indent


def maybe(method):
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ParseError:
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
    NEGATED_RELATIONAL = enum.auto()
    IF_STATEMENT = enum.auto()
    WHILE_LOOP = enum.auto()
    BREAK = enum.auto()
    CONTINUE = enum.auto()
    RANGE = enum.auto()
    FOR_ARRAY_LOOP = enum.auto()
    FOR_RANGE_LOOP = enum.auto()
    MODULE = enum.auto()
    SQUELCH = enum.auto()
    STRING = enum.auto()
    CHARACTER = enum.auto()
    TERMINATOR = enum.auto()
    IMPORT = enum.auto()
    ELLIPSIS = enum.auto()


def parser(type: NodeType):
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
            except ParseError:
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
            node.blang = node.token.text
            return node
    return None


@parser(NodeType.REF_TYPE)
def RefType(node):
    node.eat(TokenSpec.LESS_THAN)

    if not maybe(node.eat_child)(RefType) and not maybe(node.eat_child)(BaseType):
        return None
    node.eat(TokenSpec.MORE_THAN)
    node.blang = f"<{node.children[0].blang}>"
    return node


@parser(NodeType.IDENTIFIER_REF)
def IdentifierRef(node):
    node.eat(TokenSpec.MORE_THAN)
    node.eat_child(Identifier)
    node.eat(TokenSpec.LESS_THAN)
    node.blang = f">{node.children[0].blang}<"
    return node


@parser(NodeType.CHARACTER)
def Character(node):
    node.eat(TokenSpec.CHARACTER, set_leaf=True)
    node.blang = f"'{node.token.text}'"
    return node


@parser(NodeType.TERMINATOR)
def Terminator(node):
    node.eat(TokenSpec.TERMINATOR, set_leaf=True)
    node.blang = "';'"
    return node


@parser(NodeType.ARRAY)
def Array(node):
    node.eat_child(OneOf(BaseType, RefType))
    node.eat(TokenSpec.LSQBRACKET)
    node.eat_child(Integer)
    node.eat(TokenSpec.RSQBRACKET)
    node.blang = f"{node.children[0].blang}[{node.children[1].blang}]"
    return node


Type = OneOf(Array, BaseType, RefType)  # order matters


@parser(NodeType.SQUELCH)
def Squelch(node):
    node.eat_child(Type)
    node.eat(TokenSpec.BAR)
    node.eat_child(Expr)
    node.eat(TokenSpec.BAR)
    node.blang = f"{node.children[0].blang}|{node.children[1].blang}|"
    return node


@parser(NodeType.IDENTIFIER)
def Identifier(node):
    node.eat(TokenSpec.IDENTIFIER, set_leaf=True)
    node.blang = node.token.text
    node.blang = f"{node.token.text}"
    return node


@parser(NodeType.TYPED_IDENTIFER)
def TypedIdentifier(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.COLON)
    node.eat_child(Type)
    node.blang = node.children[0].blang + ": " + node.children[1].blang
    return node


@parser(NodeType.DECLARATION)
def Declaration(node):
    is_external = maybe(node.eat)(TokenSpec.EXTERN, set_child=True)
    node.eat_child(TypedIdentifier)
    node.blang = node.children[0].blang
    if not is_external and maybe(node.eat)(TokenSpec.ASSIGN):
        node.eat_child(ExprOrString)
        node.blang += " = " + node.children[-1].blang
    return node


@parser(NodeType.DE_REF)
def DeRef(node):
    node.eat(TokenSpec.LESS_THAN)
    node.eat_child(Additive)
    node.eat(TokenSpec.MORE_THAN)

    node.blang = f"<{node.children[0].blang}>"
    return node


@parser(NodeType.ARRAY_ITEM)
def ArrayItem(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.LSQBRACKET)
    node.eat_child(Additive)
    node.eat(TokenSpec.RSQBRACKET)
    node.blang = f"{node.children[0].blang}[{node.children[1].blang}]"
    return node


LVal = OneOf(DeRef, ArrayItem, Identifier)  # order matters here


@parser(NodeType.ASSIGNMENT)
def Assignment(node):
    node.eat_child(LVal)
    node.eat(TokenSpec.ASSIGN)
    node.eat_child(ExprOrString)
    node.blang = f"{node.children[0].blang} = {node.children[1].blang}"
    return node


@parser(NodeType.INTEGER)
def Integer(node):
    is_negative = maybe(node.eat)(TokenSpec.MINUS)
    node.eat(TokenSpec.INTEGER, set_leaf=True)
    if is_negative:
        node.token.text = f"-{node.token.text}"
    node.blang = node.token.text
    return node


@parser(NodeType.FLOAT)
def Float(node):
    node.eat(TokenSpec.FLOAT, set_leaf=True)
    node.blang = node.token.text
    return node


Number = OneOf(Integer, Float)


@parser(NodeType.STRING)
def String(node):
    node.eat(TokenSpec.STRING, set_leaf=True)
    node.blang = f'"{node.token.text}"'
    return node


@parser(NodeType.RETURN)
def Return(node):
    node.eat(TokenSpec.RETURN)
    node.blang = "return"
    if maybe(node.eat_child)(Expr):
        node.blang += " " + node.children[0].blang
    return node


@parser(NodeType.FUNC_CALL)
def FuncCall(node):
    node.eat_child(Identifier)
    node.eat(TokenSpec.LPAREN)
    node.blang = node.children[0].blang + "("
    while True:
        if not maybe(node.eat_child)(Expr):
            break
        node.blang += node.children[-1].blang
        if not maybe(node.eat)(TokenSpec.COMMA):
            break
        node.blang += ", "
    node.eat(TokenSpec.RPAREN)
    node.blang += ")"
    return node


@parser(NodeType.ELLIPSIS)
def Ellipsis(node):
    node.eat(TokenSpec.ELLIPSIS, set_leaf=True)
    node.blang = "..."
    return node


@parser(NodeType.PARAMETER_LIST)
def ParameterList(node):
    node.eat(TokenSpec.LPAREN)
    node.blang = "("
    while True:
        if maybe(node.eat_child)(Ellipsis):
            node.blang += "..."
            break

        if not maybe(node.eat_child)(TypedIdentifier):
            break
        node.blang += " " + node.children[-1].blang
        if not maybe(node.eat)(TokenSpec.COMMA):
            break
        node.blang += ","
    node.eat(TokenSpec.RPAREN)
    node.blang += ")"

    return node


@parser(NodeType.FUNC_DEF)
def FuncDef(node):
    node.eat(TokenSpec.DEF)
    node.eat_child(Identifier)
    node.eat_child(ParameterList)
    node.eat(TokenSpec.COLON)
    node.eat_child(Type)
    node.eat_child(Block)
    node.blang = (
        "def "
        + node.children[0].blang
        + node.children[1].blang
        + ": "
        + node.children[2].blang
    )
    return node


@parser(NodeType.FUNC_DECL)
def FuncDecl(node):
    node.eat(TokenSpec.EXTERN)
    node.eat(TokenSpec.DEF)
    node.eat_child(Identifier)
    node.eat_child(ParameterList)
    if maybe(node.eat)(TokenSpec.COLON):
        node.eat_child(Type)
    node.blang = (
        "extern def "
        + node.children[0].blang
        + node.children[1].blang
        + ": "
        + node.children[2].blang
    )
    return node


@parser(NodeType.BLOCK)
def BracedBlock(node):
    node.eat(TokenSpec.LBRACE)
    while True:
        c = maybe(node.eat_child)(Statement)
        if not c:
            break
    node.eat(TokenSpec.RBRACE)
    return node


@parser(NodeType.BLOCK)
def IndentedBlock(node):
    node.eat(TokenSpec.COLON)
    node.eat(TokenSpec.NEWLINE, ignore_space=True, ignore_linebreak=False)
    # now whatever indent comes up, use it to group block
    node.eat_child(Statement)
    indent = node.children[-1].indent_count

    while next := node.peek_child(Statement):
        if next.indent_count == indent:
            node.eat_child(Statement)
        else:
            break

    return node


Block = OneOf(BracedBlock, IndentedBlock)


@parser(NodeType.CAPTURED_EXPRESSION)
def CapturedExpression(node):
    node.eat(TokenSpec.LPAREN)
    node.eat_child(Expr)
    node.eat(TokenSpec.RPAREN)
    node.blang = "( " + node.children[0].blang + ")"
    return node.children[0]


@parser(NodeType.BOOLEAN)
def Boolean(node):
    if maybe(node.eat)(TokenSpec.TRUE, set_leaf=True) or maybe(node.eat)(
        TokenSpec.FALSE, set_leaf=True
    ):
        node.blang = node.token.text
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
    node.blang = node.children[0].blang
    while maybe(node.eat)(TokenSpec.MINUS, set_leaf=True) or maybe(node.eat)(
        TokenSpec.PLUS, set_leaf=True
    ):
        node.blang += " " + node.token.text + " "
        node.eat_child(Term)
        node.blang += node.children[-1].blang
        left = copy.deepcopy(node)
        node.children = []
        node.children.append(left)

    if len(node.children) == 1:
        return node.children[0]
    return node


@parser(NodeType.TERM)
def Term(node):
    node.eat_child(Factor)
    node.blang = node.children[0].blang
    while (
        maybe(node.eat)(TokenSpec.ASTRISK, set_leaf=True)
        or maybe(node.eat)(TokenSpec.DIVIDE, set_leaf=True)
        or maybe(node.eat)(TokenSpec.MODULO, set_leaf=True)
    ):
        node.eat_child(Factor)
        node.blang += node.token.text
        node.blang += node.children[-1].blang
        left = copy.deepcopy(node)
        node.children = []
        node.children.append(left)

    if len(node.children) == 1:
        return node.children[0]
    return node


@parser(NodeType.RELATIONAL)
def Relational(node):
    node.eat_child(Additive)
    node.blang = node.children[0].blang
    if (
        maybe(node.eat)(TokenSpec.MORE_THAN, set_leaf=True)
        or maybe(node.eat)(TokenSpec.LESS_THAN, set_leaf=True)
        or maybe(node.eat)(TokenSpec.MORE_THAN_EQ, set_leaf=True)
        or maybe(node.eat)(TokenSpec.LESS_THAN_EQ, set_leaf=True)
        or maybe(node.eat)(TokenSpec.EQUAL, set_leaf=True)
        or maybe(node.eat)(TokenSpec.NOT_EQ, set_leaf=True)
    ):
        node.eat_child(Additive)
        node.blang += " " + node.token.text + node.children[1].blang
        return node
    return node.children[0]


@parser(NodeType.LOGIC_AND)
def LogicAnd(node):
    node.eat_child(OneOf(Relational, NegatedRelational))
    node.blang = node.children[0].blang
    if maybe(node.eat)(TokenSpec.AND):
        node.eat_child(OneOf(Relational, NegatedRelational))
        node.blang += " and " + node.children[1].blang
        return node
    return node.children[0]


@parser(NodeType.LOGIC_OR)
def LogicOr(node):
    node.eat_child(LogicAnd)
    node.blang = node.children[0].blang
    if maybe(node.eat)(TokenSpec.OR):
        node.eat_child(LogicAnd)
        node.blang += " or " + node.children[1].blang
        return node
    return node.children[0]


@parser(NodeType.NEGATED_RELATIONAL)
def NegatedRelational(node):
    node.eat(TokenSpec.NOT)
    node.eat_child(Relational)
    node.blang = "not " + node.children[0].blang
    return node


@parser(NodeType.IF_STATEMENT)
def IfStatement(node):
    node.eat(TokenSpec.IF)
    node.eat_child(Expr)
    node.eat_child(Block)

    node.blang = "if " + node.children[0].blang + " (" + node.children[1].blang + ")"
    if maybe(node.eat)(TokenSpec.ELSE):
        node.eat_child(Block)
    return node


@parser(NodeType.WHILE_LOOP)
def WhileLoop(node):
    node.eat(TokenSpec.WHILE)
    node.eat_child(Expr)
    node.eat_child(Block)
    node.blang = "while " + node.children[0].blang

    return node


@parser(NodeType.BREAK)
def Break(node):
    node.eat(TokenSpec.BREAK)
    node.blang = "break"
    return node


@parser(NodeType.CONTINUE)
def Continue(node):
    node.eat(TokenSpec.CONTINUE)
    node.blang = "continue"
    return node


@parser(NodeType.FOR_ARRAY_LOOP)
def ForArrayLoop(node):
    node.eat(TokenSpec.FOR)
    node.eat_child(Identifier)
    node.eat(TokenSpec.AS)
    node.eat_child(Identifier)
    node.eat(TokenSpec.COMMA)
    node.eat_child(Identifier)
    node.eat_child(Block)
    node.blang = f"for {node.children[0].blang} as {node.children[1].blang}, {node.children[1].blang}"
    return node


@parser(NodeType.FOR_RANGE_LOOP)
def ForRangeLoop(node):
    node.eat(TokenSpec.FOR)
    node.eat_child(Range)
    node.eat(TokenSpec.AS)
    node.eat_child(TypedIdentifier)
    node.eat_child(Block)
    node.blang = f"for {node.children[0].blang} as {node.children[1].blang}"

    return node


@parser(NodeType.RANGE)
def Range(node):
    node.eat_child(Expr)
    node.eat(TokenSpec.DOTDOT)
    node.eat_child(Expr)
    if maybe(node.eat)(TokenSpec.COLON):
        node.eat_child(Expr)
    node.blang = f"{node.children[0].blang}..{node.children[1].blang}" + (
        f": {node.children[2].blang}" if len(node.children) > 2 else ""
    )
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
    Continue,
    Block,
    Terminator,  # a no op
)


@parser(NodeType.IMPORT)
def Import(node):
    node.eat(TokenSpec.IMPORT)
    node.blang = "import "
    # dot separated path
    if maybe(node.eat)(TokenSpec.DOT, set_child=True):
        node.blang += "."

    node.eat_child(Identifier)
    node.blang += node.children[-1].token.text
    while maybe(node.eat)(TokenSpec.DOT):
        node.eat_child(Identifier)
        node.blang += "." + node.children[-1].token.text

    return node


DecOrDef = OneOf(FuncDef, FuncDecl, Declaration, Terminator, Import)


@parser(NodeType.MODULE)
def Module(node):
    while maybe(node.eat_child)(DecOrDef):
        continue
    # ignore white space at the end too
    while maybe(node.eat)(TokenSpec.WHITESPACE) or maybe(node.eat)(TokenSpec.NEWLINE):
        continue
    if node._eaten != len(node._tokens):
        print(f"UNEXPECTED TOKEN '{node._tokens[node._eaten].text}'")
        print("_tokens=", node._tokens)
        print("type=", node.type)
        print("Line number:", node._tokens[node._eaten].lineno)
        raise ParseError(f"unexpected token {node._tokens[node._eaten]}", node)
    return node
