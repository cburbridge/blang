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
        return f"line {self.node._tokens[0].lineno} \n" + self.message


_node_compilers = {}


def node_compiler(type=NodeType.UNKNOWN):
    def node_compiler(method):
        def _wrap(*args, **kwargs):
            return method(*args, **kwargs)

        _node_compilers[type] = _wrap
        return _wrap

    return node_compiler


def compile(node, context):
    try:
        asm = _node_compilers[node.type](node, context)
    except AttributeError as e:
        print_tree(node)
        raise CompileError(f"Bad times. {e}", node)
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
    var = typed_identifier_to_variable(node.children[0])
    init = None
    if len(node.children) > 1:
        init = node.children[1]

    if not context.use_stack:
        if init and init.type not in (NodeType.FLOAT, NodeType.INTEGER):
            raise CompileError("Can't initialise with non-constant.", init)

        if context.is_local_var(var.identifier):
            raise CompileError(
                f"Variable '{var.identifier}' is already defined.", node.children[0]
            )

        var.location = f"[{var.identifier}]"
        context.globals_[var.identifier] = var

        if init:
            dd = SizeDefiners[var.size]
            return [f"{var.identifier}: {dd} {init.token.text}"]

        res = SizeReserves[var.size]
        return [f"{var.identifier}: {res} 1"]
    else:
        if context.is_local_var(var.identifier):
            raise CompileError(
                f"Variable '{var.identifier}' is already defined.", node.children[0]
            )

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
    context.current_func = function
    # Set the parameters as locals
    context.new_frame()
    rbp_offset = 16  # todo why this? stack params are broke
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
        f"global {function.identifier}",  # if function.exported
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
        print(child)
        asm.extend(compile(child, context))
    context.pop_frame()
    return asm


@node_compiler(NodeType.IDENTIFIER_REF)
def compile_ref(node, context: Context):
    # lea rax, [q]        ; Load address of q into rax
    var_name = node.children[0].token.text
    var = context.variables.get(var_name)
    if not var:
        raise CompileError("Unknown variable '{var_name}'", node.children[0])

    reg = context.free_registers.pop()
    reg.type = var.type
    reg.indirection_count = var.indirection_count + 1
    context.occupied_registers.append(reg)
    asm = [f"lea {reg}, {var.location}"]
    return asm


@node_compiler(NodeType.DE_REF)
def compile_de_ref(node, context: Context):
    # note, this is compiling only a an rval, in assignment this is not run
    ptr, code = compile_to_literal(node.children[0], context)
    if ptr in context.occupied_registers:  # its a register so use it
        reg = ptr
    else:
        reg = context.free_registers.pop()
        context.occupied_registers.append(reg)
        reg.type = ptr.type
        reg.indirection_count = ptr.indirection_count
        code = [
            *code,
            f"mov {reg.full_reg}, {ptr}",
        ]

    if reg.indirection_count < 1:
        raise CompileError("Problem. Can't dereference a non-reference.", node)
    reg.indirection_count -= 1
    asm = [*code]  # f"mov qword {reg}, {reg.location}"]
    # use the same register, drop size
    asm.append(
        f"mov {SizeSpecifiers[reg.size]}  {reg}, [{reg.full_reg}]; ;cnt={reg.indirection_count}"
    )
    return asm


def compile_to_literal(
    node: Node, context: Context
) -> tuple[Literal | Register | Variable, str]:
    # to an identifier or a register or literal
    if node.typ in (NodeType.IDENTIFIER,):
        if node.token.text not in context.variables:
            raise CompileError(f"Unknown variable {node.token.text}", node)
        return context.variables[node.token.text], []
    if node.typ in (NodeType.INTEGER):  # todo literal sizes and types
        return Literal(node.token.text, type=VariableType.u32), []

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
        reg.indirection_count = l.indirection_count
        reg.type = l.type
        context.occupied_registers.append(reg)

    if l != reg and l in context.occupied_registers:  # todo dead?
        context.occupied_registers.remove(l)
        context.free_registers.append(l)

    r, code_r = compile_to_literal(node.children[1], context)
    if not isinstance(r, Literal) and r.size != reg.size:
        raise CompileError(
            f"Size mismatch. {r.identifier} is {r.size} bytes, {l.identifier} is {l.size} bytes. May need to squelch.",
            node,
        )
    if r.type != reg.type or (r.indirection_count > 1 != reg.indirection_count > 1):
        print(
            "WARNING: Type mismatch in addition. Sizes match so proceeding. May be bad."
        )

    if r in context.occupied_registers:
        context.occupied_registers.remove(r)
        context.free_registers.append(r)

    ref_muliplier = 1
    if reg.indirection_count:
        reg.indirection_count -= 1
        ref_muliplier = reg.size
        reg.indirection_count += 1

    return [
        *code_l,
        *code_r,  #
        *((f"mov {reg}, {l}",) if l != reg else ()),
        *(f"{op} {reg}, {r}" for _ in range(ref_muliplier)),
    ]


@node_compiler(NodeType.ASSIGNMENT)
def compile_assignment(node, context: Context):
    # compile what will be assigned to it
    reg, asm = compile_to_literal(node.children[1], context)

    if node.children[0].type == NodeType.DE_REF:
        decount = 1
        id_node = node.children[0].children[0]
        while id_node.type == NodeType.DE_REF:
            decount += 1
            id_node = id_node.children[0]
        lval, get_addr_asm = compile_to_literal(id_node, context)

        if lval.indirection_count < decount:
            raise CompileError("Tried to deref too many times in lval.", node)

        lval.indirection_count -= decount
        sizespec = SizeSpecifiers[lval.size]
        lval.indirection_count += decount
        if lval not in context.occupied_registers:
            # not a register, need to put it in one to be able to deref in x86
            target_reg = context.free_registers[0]
            target_reg.set_in_use(8)
            load_value = [
                f"mov {target_reg.full_reg}, {lval.location}",
            ]
        else:
            target_reg = lval
            load_value = []

        # need to deref decount-1 more times now
        load_value.extend(
            (f"mov {target_reg.full_reg}, [{target_reg.full_reg}]",) * (decount - 1)
        )

        target_address = target_reg.full_reg

        return [
            *asm,
            *get_addr_asm,
            *load_value,
            f"mov {sizespec} [{target_address}], {reg}",  # assign the value
        ]

    else:
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
        rax.type = reg.type
        rax.indirection_count = reg.indirection_count
        asm = (*asm, "xor rax, rax", f"mov {rax}, {reg}")
    else:
        asm = []
    return [*asm, f"jmp {context.current_func.identifier}___ret"]


@node_compiler(NodeType.FUNC_CALL)
def compile_call(node, context: Context):
    function_name = node.children[0].token.text
    try:
        function = context.variables[function_name]
        assert isinstance(function, Function)
    except:
        raise CompileError(f"No function {function_name}.", node)

    param_fill_asm = []
    for parameter, i, parameter_decl in zip(
        node.children[1:], range(len(function.parameters)), function.parameters
    ):
        target_register = ArgumentRegistersBySize[parameter_decl.size][i]
        p_reg, asm = compile_to_literal(parameter, context)
        asm.append(
            f"mov {SizeSpecifiers[parameter_decl.size]} {target_register}, {p_reg}"
        )
        param_fill_asm.extend(asm)
        context.mark_free_if_reg(p_reg)

    return_val = Register("rax")
    return_val.set_in_use(function.size)
    return_val.type = function.type
    return_val.indirection_count = function.indirection_count
    context.occupied_registers.append(return_val)
    return [*param_fill_asm, f"call {function_name}"]


@node_compiler(NodeType.IF_STATEMENT)
def compile_if(node, context: Context):
    return compile_block(node.children[1], context)


def compiler(text):
    tokens = list(TokenSpec.tokenise(text))
    # for p in tokens:
    # print(p.typ, p.text)
    module = Module(tokens)
    if not module:
        return None
    print_tree(module)
    return compile(module, Context())
