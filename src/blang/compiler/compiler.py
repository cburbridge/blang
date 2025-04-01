from blang.parser import Module, print_tree, NodeType, Node, Expr
from blang import parser
from blang.tokeniser import TokenSpec

# from blang.compiler.types import *
from blang.compiler.types import (
    Literal,
    Register,
    Variable,
    Function,
    Context,
    ArgumentRegistersBySize,
    TokenVariableTypeMap,
    VariableType,
    SizeSpecifiers,
    SizeReserves,
    SizeDefiners,
)


class CompileError(Exception):
    def __init__(self, message, node):
        self.message = message
        self.node = node

    def __str__(self):
        return f"line {self.node.token.lineno} \n" + self.message


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
    return asm


def type_to_base_type_and_indirection_count(type) -> (VariableType, int):
    refs = 0
    while type.typ == NodeType.REF_TYPE:
        type = type.children[0]
        refs += 1
    return TokenVariableTypeMap[type.token.typ], refs


def typed_identifier_to_variable(typed_identifier: Node):
    identifier, type = typed_identifier.children
    type, refs = type_to_base_type_and_indirection_count(type)

    return Variable(identifier=identifier.token.text, type=type, indirection_count=refs)


@node_compiler(parser.NodeType.MODULE)
def compile_module(node, context: Context) -> str:
    asm = ""
    # Collect declarations
    declarations = list(
        filter(lambda x: x.type == parser.NodeType.DECLARATION, node.children)
    )
    asm = []
    asm.append("section .data")
    for declaration in filter(lambda d: len(d.children) > 1, declarations):
        asm.extend(compile(declaration, context))
    asm.append("")
    asm.append("section .bss")
    for declaration in filter(lambda d: len(d.children) < 2, declarations):
        asm.extend(compile(declaration, context))
    asm.append(" ")

    # Collect funcs
    functions = list(
        filter(lambda x: x.type == parser.NodeType.FUNC_DEF, node.children)
    )
    # pass 1 to get function declarations into globals
    for function in functions:
        identifier, parameters, type = function.children[:3]
        type, refs = type_to_base_type_and_indirection_count(type)
        f = Function(
            identifier=identifier.token.text,
            type=type,
            indirection_count=refs,
            parameters=[
                typed_identifier_to_variable(t)
                for t in parameters.children
                if t.children
            ],
        )
        if f.identifier in context.variables:
            raise CompileError(f"Duplicate declaration of '{f.identifier}'", identifier)
        context.globals_[f.identifier] = f

    # pass 2 to compile definitions
    asm.append("section .text")
    for function in functions:
        asm.extend(compile(function, context))
    return asm


@node_compiler(parser.NodeType.DECLARATION)
def compile_declaration(node, context: Context):
    print_tree(node)
    var = typed_identifier_to_variable(node.children[0])
    init = None
    if len(node.children) > 1:
        init = node.children[1]

    if not context.use_stack:
        if init and init.type not in (NodeType.FLOAT, NodeType.INTEGER):
            raise CompileError("Can't initialise with non-constant.", init)

        if context.is_local_var(var.identifier):
            raise CompileError(f"Variable '{var.identifier}' is already defined.", node)

        var.location = f"[{var.identifier}]"
        context.globals_[var.identifier] = var

        if init:
            dd = SizeDefiners[var.size]
            return [f"{var.identifier}: {dd} {init.token.text}"]

        res = SizeReserves[var.size]
        return [f"{var.identifier}: {res} 1"]
    else:
        if context.is_local_var(var.identifier):
            raise CompileError(f"Variable '{var.identifier}' is already defined.", node)

        context.locals_stack_size += var.size
        var.location = f"[rbp - {context.locals_stack_size}]"
        context.variables[var.identifier] = var
        initialise = ()
        if init:
            sizespec = SizeSpecifiers[var.size]
            reg, asm = compile_to_literal(init, context)
            initialise = (*asm, f"mov {sizespec} {var.location}, {reg}")

        return [f"; {var.identifier} @ {var.location}", *initialise]


@node_compiler(NodeType.FUNC_DEF)
def comile_func(node, context: Context):
    # we already created the Function object in the globals as part of the module
    # so we can reuse it
    identifier, parameters, type, block = node.children
    function: Function = context.variables.get(identifier.token.text)
    if not function:
        # we could create it here to make sure to support compile this without modile compile
        # assumptions. todo
        ...

        raise NotImplementedError(
            "Need to make funcs compile not relying on module to create  declaration."
        )
    context.locals_stack_size = 0
    context.use_stack = True
    context.current_func = function.identifier
    # Set the parameters as locals
    context.new_frame()
    rbp_offset = 16
    for i, parameter in enumerate(function.parameters):
        if i > 5:
            location = f"[rbp + {rbp_offset}]"
            rbp_offset += parameter.size
        else:
            location = ArgumentRegistersBySize[parameter.size][i]

        context.variables[parameter.identifier] = Variable(
            parameter.identifier,
            location=location,
            type=parameter.type,
            indirection_count=parameter.indirection_count,
            on_stack=False,
        )

    blk = compile(block, context)
    context.pop_frame()
    context.current_func = None
    return [
        "",
        f"{function.identifier}:",
        "push rbp",
        "mov rbp, rsp",
        f"sub rsp, {context.locals_stack_size}",
        *blk,
        f"{function.identifier}___ret:",
        "leave",
        "ret",
    ]


@node_compiler(NodeType.BLOCK)
def compile_block(node, context: Context):
    asm = []
    context.new_frame()
    for child in node.children:
        asm.extend(compile(child, context))
    context.pop_frame()
    return asm


def compile_to_literal(
    node: Node, context: Context
) -> tuple[Literal | Register | Variable, str]:
    # to an identifier or a register or literal
    if node.typ in (NodeType.IDENTIFIER,):
        if node.token.text not in context.variables:
            raise CompileError(f"Unknown variable {node.token.text}", node)
        return context.variables[node.token.text], []
    if node.typ in (NodeType.INTEGER):  # todo literal sizes
        return Literal(node.token.text, 4), []

    asm = compile(node, context)
    try:
        # last taken occupied is the reg that compiled into
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
    op = "add" if node.token == TokenSpec.PLUS else "sub"
    l, code_l = compile_to_literal(node.children[0], context)

    if l in context.occupied_registers:  # its a register so use i
        reg = l
    else:
        reg = context.free_registers.pop()
        reg.set_in_use(l.size)
        context.occupied_registers.append(reg)

    if l != reg and l in context.occupied_registers:
        context.occupied_registers.remove(l)
        context.free_registers.append(l)

    r, code_r = compile_to_literal(node.children[1], context)
    if r.size != reg.size:
        raise CompileError(
            f"Size mismatch. {r} is {r.size} bytes, {l} is {l.size} bytes. May need to squelch.",
            node,
        )

    if r in context.occupied_registers:
        context.occupied_registers.remove(r)
        context.free_registers.append(r)

    return [
        *code_l,
        *code_r,  #
        *((f"mov {reg}, {l}",) if l != reg else ()),
        f"{op} {reg}, {r}",
    ]


@node_compiler(NodeType.ASSIGNMENT)
def compile_assignment(node, context: Context):
    reg, asm = compile_to_literal(node.children[1], context)
    var = context.variables[node.children[0].token.text]
    sizespec = SizeSpecifiers[var.size]
    return [*asm, f"mov {sizespec} {var.location}, {reg}"]


@node_compiler(NodeType.RETURN)
def compile_return(node, context: Context):
    if len(node.children) == 1:
        ret = node.children[0]
        reg, asm = compile_to_literal(ret, context)
        rax = Register("rax")
        rax.set_in_use(reg.size)
        asm = (*asm, "xor rax, rax", f"mov {rax}, {reg}")
    else:
        asm = []
    return [*asm, f"jmp {context.current_func}___ret"]


@node_compiler(NodeType.FUNC_CALL)
def compile_call(node, context: Context):
    function_name = node.children[0].token.text
    try:
        function = context.variables[function_name]
        assert isinstance(function, Function)
    except:
        raise CompileError(f"No function {function_name}.", node)

    print(f"Func {function.identifier} : params {function.parameters}")
    print_tree(node)
    param_fill_asm = []
    for parameter, i, parameter_decl in zip(
        node.children[1:], range(len(function.parameters)), function.parameters
    ):
        target_register = ArgumentRegistersBySize[parameter_decl.size][i]
        p_reg, asm = compile_to_literal(parameter, context)
        print(f" -> {parameter.token.text}\n********\n")
        print(f"{p_reg}  \n{asm}")
        print(f"Put it into {target_register}")
        asm.append(
            f"mov {SizeSpecifiers[parameter_decl.size]} {target_register}, {p_reg}"
        )
        param_fill_asm.extend(asm)
        context.mark_free_if_reg(p_reg)

    return_val = Register("rax")
    return_val.set_in_use(function.size)
    context.occupied_registers.append(return_val)
    return [*param_fill_asm, f"call {function_name}"]


def compiler(text):
    tokens = list(TokenSpec.tokenise(text))
    # for p in tokens:
    # print(p.typ, p.text)
    module = Module(tokens)
    # print_tree(module)
    return compile(module, Context())


def test_dev():
    """ """
    program = """    
    a: u32
    b: u8 = 99
    c: u32
    d: u32

    def add(p:u32, q:u32):u32 {
      a : <<u32>>
      return p + q
    }

    def pt_add(t:u8):u8 {
        q:<u8> = 100
        return t
    }
    
    def main(): u32 {
        v: u32 = 5
        q: u32 = 5
        d = 10
        s: u32 = 10
        t: u32 = 5
        PTR: <u8> = 0
        tt: u32 = add(5,10)
        return v+q+s-t + d+tt
    } 
    """
    asm = compiler(program)
    asm.extend(
        [
            "",
            "global _start",
            "; --- ENTRY POINT ---",
            "_start:",
            "    call main          ; call main()",
            "",
            "    mov edi, eax       ; move main's return value to sys_exit arg",
            "    mov eax, 60        ; sys_exit syscall",
            "    syscall",
        ]
    )
    print("\n".join(asm))
    with open("/home/chris/code/blang/first.asm", "w") as f:
        f.write("\n".join(asm))
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
