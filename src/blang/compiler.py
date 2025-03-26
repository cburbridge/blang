from .parser import Module, print_tree, NodeType, Node, Expr
from . import parser
from .tokeniser import TokenSpec

from collections import defaultdict
from dataclasses import dataclass, field
import enum
import textwrap


class CompileError(Exception):
    def __init__(self, message, node):
        self.message = message
        self.node = node

    def __str__(self):
        return f"line {self.node.token.lineno} \n" + self.message


class VariableType(enum.StrEnum):
    u8 = enum.auto()
    u16 = enum.auto()
    u32 = enum.auto()
    u64 = enum.auto()
    u128 = enum.auto()
    i8 = enum.auto()
    i16 = enum.auto()
    i32 = enum.auto()
    i64 = enum.auto()
    i128 = enum.auto()
    flt = enum.auto()


TokenVariableTypeMap = {
    TokenSpec.U8: VariableType.u8,
    TokenSpec.U16: VariableType.u16,
    TokenSpec.U32: VariableType.u32,
    TokenSpec.U64: VariableType.u64,
    TokenSpec.U128: VariableType.u128,
    TokenSpec.I8: VariableType.i8,
    TokenSpec.I16: VariableType.i16,
    TokenSpec.I32: VariableType.i32,
    TokenSpec.I64: VariableType.i64,
    TokenSpec.I128: VariableType.i128,
    TokenSpec.FLOAT: VariableType.flt,
}

TypeSizes = {
    VariableType.u8: 1,
    VariableType.u16: 2,
    VariableType.u32: 4,
    VariableType.u64: 8,
    VariableType.u128: 16,
    VariableType.i8: 1,
    VariableType.i16: 2,
    VariableType.i32: 4,
    VariableType.i64: 8,
    VariableType.i128: 16,
    VariableType.flt: 8,
}

SizeReserves = {1: "resb", 2: "resw", 4: "resd", 8: "resq", 16: "resdq"}

SizeDefiners = {1: "db", 2: "dw", 4: "dd", 8: "dq", 16: "do"}

ArgumentRegistersBySize = {
    8: ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
    4: ["edi", "esi", "edx", "ecx", "r8d", "r9d"],
    2: ["di", "si", "dx", "cx", "r8w", "r9w"],
    1: ["dil", "sil", "dxl", "cxl", "r8b", "r9b"],
}

_node_compilers = {}


def node_compiler(type=NodeType.UNKNOWN):
    def node_compiler(method):
        def _wrap(*args, **kwargs):
            return method(*args, **kwargs)

        _node_compilers[type] = _wrap
        return _wrap

    return node_compiler


def compile(node, context):
    asm = _node_compilers[node.type](node, context)
    print("--" * 10)
    print(node.type)
    print(asm)
    return asm


@dataclass
class Variable:
    identifier: str
    location: str
    type: VariableType
    on_stack: bool
    external: bool = False
    exported: bool = False
    node: Node = None


@dataclass
class Function:
    identifier: str
    type: VariableType
    parameters: list[VariableType]
    external: bool
    exported: bool
    node: Node | None = None


@dataclass
class Context:
    variable_stack: list[dict] = field(default_factory=lambda: [dict()])
    locals_stack_size: int = 0
    use_stack: bool = False
    current_func: str = None
    free_registers: list = field(
        default_factory=lambda: [
            "r10",
            "r11",
            "r12",
            "r13",
            "r14",
            "r15",
        ]
    )
    occupied_registers: list = field(default_factory=lambda: [])

    @property
    def globals_(self):
        return self.variable_stack[0]

    @property
    def locals_(self):
        return self.variable_stack[-1]

    def pop_locals():
        assert len(self.variable_stack) > 2
        self.variable_stack = self.variable_stack[:-1]

    def new_frame(self):
        self.variable_stack.append({})


@node_compiler(parser.NodeType.MODULE)
def compile_module(node, context: Context) -> str:
    asm = ""
    # Collect declarations
    declarations = list(
        filter(lambda x: x.type == parser.NodeType.DECLARATION, node.children)
    )
    asm = ""
    for declaration in declarations:
        asm += compile(declaration, context) + "\n"

    # Collect funcs
    functions = list(
        filter(lambda x: x.type == parser.NodeType.FUNC_DEF, node.children)
    )
    # pass 1 to get declarations
    for function in functions:
        identifier, parameters, type = function.children[:3]
        print(parameters)
        f = Function(
            identifier.token.text,
            type=TokenVariableTypeMap[type.token.typ],
            parameters=[
                TokenVariableTypeMap[t.children[1].token.typ]
                for t in parameters.children
                if t.children
            ],
            external=False,
            exported=False,
        )
        if f.identifier in context.globals_ or f.identifier in context.locals_:
            raise CompileError(f"Duplicate declaration of '{f.identifier}'", identifier)
        context.globals_[f.identifier] = f
    # pass 2 to compile definitions
    for function in functions:
        asm += compile(function, context) + "\n"
    return asm


@node_compiler(parser.NodeType.DECLARATION)
def compile_declaration(node, context: Context):
    identifier, type = node.children[0].children
    type = TokenVariableTypeMap[type.token.typ]
    init = None
    if len(node.children) > 1:
        init = node.children[1]

    if not context.use_stack:
        if init and init.type not in (NodeType.FLOAT, NodeType.INTEGER):
            raise CompileError("Can't initialise with non-constant.", init)

        if (
            identifier.token.text in context.globals_
            or identifier.token.text in context.locals_
        ):
            raise CompileError(
                f"Variable '{identifier.token.text}' is already defined.", identifier
            )

        var = Variable(
            identifier.token.text,
            location=f"[{identifier.token.text}]",
            type=type,
            on_stack=False,
            external=False,
            exported=False,
        )
        context.globals_[var.identifier] = var

        if init:
            dd = SizeDefiners[TypeSizes[type]]
            return f"section .data\n{identifier.token.text}: {dd} {init.token.text}"

        res = SizeReserves[TypeSizes[type]]
        return f"section .bss\n{identifier.token.text}: {res} 1"
    else:
        if identifier.token.text in context.locals_:
            raise CompileError(
                f"Variable '{identifier.token.text}' is already defined.", identifier
            )

        context.locals_stack_size += TypeSizes[type]  # what
        var = Variable(
            identifier.token.text,
            location=f"[rbp - {context.locals_stack_size}]",
            type=type,
            on_stack=False,
            external=False,
            exported=False,
        )
        context.locals_[var.identifier] = var

        return f"; {var.identifier} @ {var.location} .. {init}"


@node_compiler(NodeType.FUNC_DEF)
def comile_func(node, context: Context):
    identifier, parameters, type, block = node.children
    parameters = list(
        (t.children[0].token.text, TokenVariableTypeMap[t.children[1].token.typ])
        for t in parameters.children
        if t.children
    )
    print(parameters)
    context.locals_stack_size = 0
    context.use_stack = True
    context.current_func = identifier.token.text
    # Set the parameters as locals
    context.new_frame()
    rbp_offset = 16
    for i, (param_name, param_type) in enumerate(parameters):
        if i > 5:
            location = f"[rbp + {rbp_offset}]"
            rbp_offset += TypeSizes[param_type]
        else:
            location = ArgumentRegistersBySize[TypeSizes[param_type]][i]

        param_type = VariableType[param_type]
        context.locals_[param_name] = Variable(
            param_name,
            location=location,
            type=param_type,
            on_stack=False,
        )

    print(context.locals_)
    blk = compile(block, context)
    return (
        f"{identifier.token.text}:",
        "push rbp",
        "mov rbp, rsp",
        "sub rsp, {context.locals_stack_size}",
        *blk,
        f"""{context.current_func}___ret:",
      "leave",
      "ret"
    """,
    )


@node_compiler(NodeType.BLOCK)
def compile_block(node, context: Context):
    # declarations = list(
    # filter(lambda x: x.type == parser.NodeType.DECLARATION, node.children)
    # )
    asm = ""
    # for declaration in declarations:
    # asm += compile(declaration, context) + "\n"
    for child in node.children:
        asm += compile(child, context) + "\n"
    return asm


tokies = list("abcdef")
useies = []


def compile_to_literal(node: Node, context: Context):
    # to and identifier or a register( or a number but no )
    if node.typ in (NodeType.IDENTIFIER,):
        return f"[{node.token.text}]", ""
    if node.typ in (NodeType.INTEGER):
        return node.token.text, ""
    # take a register
    # reg = context.free_registers[-1]
    asm = compile(node, context)
    try:
        reg = context.occupied_registers[-1]
    except:
        raise

    return reg, asm


def test_compile_to_lit():
    program = """(a + b - c + 5 + N - Q + a + b)+ (d + e+1)    """
    tokens = list(TokenSpec.tokenise(program))
    ex = Expr(tokens)
    print_tree(ex)
    context = Context()
    reg, code = compile_to_literal(ex, context)
    print("---- CODE ----")
    print("\n".join(code))
    print("-------------")
    print(f"going into {reg}")
    print(context)
    assert False, "Forced fail."


@node_compiler(NodeType.ADDITIVE)
def compile_additive(node: Node, context: Context):
    prem = []

    reg = context.free_registers.pop()
    context.occupied_registers.append(reg)

    op = "add" if node.token == TokenSpec.PLUS else "sub"
    l, code_l = compile_to_literal(node.children[0], context)

    if l in context.occupied_registers:  # its a register so use i
        context.occupied_registers.remove(reg)
        context.free_registers.append(reg)
        reg = l

    if l != reg and l in context.occupied_registers:
        context.occupied_registers.remove(l)
        context.free_registers.append(l)

    r, code_r = compile_to_literal(node.children[1], context)

    if r in context.occupied_registers:
        context.occupied_registers.remove(r)
        context.free_registers.append(r)

    return (
        *prem,
        *code_l,
        *code_r,  #
        *((f"mov {reg}, {l}",) if l != reg else ()),
        f"{op} {reg}, {r}",
    )


@node_compiler(NodeType.ASSIGNMENT)
def compile_assignment(node, context: Context):
    return ["mov some to some"]


@node_compiler(NodeType.RETURN)
def compile_return(node, context: Context):
    return [f"jmp {context.current_func}___ret"]


@node_compiler(NodeType.FUNC_CALL)
def compile_call(node, context: Context):
    return ["call"]


def compiler(text):
    tokens = list(TokenSpec.tokenise(text))
    for p in tokens:
        print(p.typ, p.text)
    module = Moudule(tokens)
    print_tree(module)
    return compile(module, Context())


def test_dev():
    program = """
        a: u32
        b: u64 = 99
        c: u32

        def add(p:u32, q:u64):u32 {
          l: u32
          v: u32
          v = 100
          return p*q*v
        } 
        def main() :u8 {
           add(9,10)
           return 0
        } 
    """
    print(compiler(program))
    assert False, "Forced fail."


def test_dev_long():
    program = """
    a:u32
    def fn():u8{
        a: u32
        b: <<u32>>
        q: u32[10]
        for 10 q as i, n {
           n = i
        }
        a = 1
        b = >a<
        c: u32 = a * <b+5-6+10-c*3>
        d: i32 = 0-1
        b: bool = false and true
        b = 54 < (6+100 > 10)

        do_it(<d>)
    }
    
    def do_it(i:<u32>) : <u32> {
      if (i>10) {
         i=0
      }
      while (i<10) {
        i = i + 1
        if i >3 {break}
      }
      return < >i< + 10 >
    }
    
    """
    compile(program, Context())
    assert False, "Forced fail."
