from pathlib import Path

from blang.parser import Module, print_tree, NodeType, Node
from blang import parser
from blang.tokeniser import TokenSpec
import os

from blang.compiler.types import (
    Literal,
    Register,
    Variable,
    Array,
    Function,
    Context,
    ArgumentRegistersBySize,
    TokenVariableTypeMap,
    VariableType,
    SizeSpecifiers,
    SizeReserves,
    TypeSizes,
    SizeDefiners,
    AddressPointer,
    RBP,
    SignedIntVariableTypes,
)


class CompileError(Exception):
    def __init__(self, message, node):
        self.message = message
        self.node = node

    def __str__(self):
        return f"line {self.node._tokens[0].lineno} \n" + self.message


_node_compilers = {}


def node_compiler(type: NodeType):
    def node_compiler(method):
        def _wrap(*args, **kwargs):
            return method(*args, **kwargs)

        _node_compilers[type] = _wrap
        return _wrap

    return node_compiler


def compile(node, context):
    try:
        asm, registers = _node_compilers[node.type](node, context)

        if asm:
            spacing = max(30 - len(asm[0]), 0)
            asm[0] += spacing * " " + f"; L{node._tokens[0].lineno}: " + node.blang
    except AttributeError as e:
        print_tree(node)
        raise CompileError(f"Bad times. {e}", node)
    return asm, registers


def type_to_base_type_and_indirection_count(type) -> (VariableType, int):
    refs = 0
    while type.typ == NodeType.REF_TYPE:
        type = type.children[0]
        refs += 1
    return TokenVariableTypeMap[type.token.typ], refs


def typed_identifier_to_variable(typed_identifier: Node):
    identifier, type = typed_identifier.children
    array_size = 0
    if type.type == NodeType.ARRAY:
        array_size = int(type.children[1].token.text)
        type = type.children[0]

    type, refs = type_to_base_type_and_indirection_count(type)

    if array_size > 0:
        return Array(
            identifier=identifier.token.text,
            type=type,
            indirection_count=refs,
            length=array_size,
        )

    return Variable(identifier=identifier.token.text, type=type, indirection_count=refs)


def collect_functions_from_module(module: Node, context: Context):
    """Extracts function declarations from a module and adds Function objects
    for them to the context.variables. Returns list of added functions."""
    functions = []
    # pass 1 to get function declarations into globals
    for function in list(
        filter(
            lambda x: x.type in (parser.NodeType.FUNC_DEF, parser.NodeType.FUNC_DECL),
            module.children,
        )
    ):
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
            node=function,
        )
        if f.identifier in context.variables:
            raise CompileError(f"Duplicate declaration of '{f.identifier}'", identifier)
        context.globals_[f.identifier] = f
        functions.append(f)
    return functions


def import_node_to_filepath(node, imported_from: Path):
    if node.children[0].type == TokenSpec.DOT:
        filename = imported_from.parent / os.path.join(
            *(n.token.text for n in node.children)
        )
    else:
        filename = Path("/") / os.path.join(*(n.token.text for n in node.children))
    return filename.with_suffix(".bl")


@node_compiler(parser.NodeType.IMPORT)
def compile_import(node, context: Context) -> str:
    asm = []
    filename = import_node_to_filepath(node, context.filename)
    print(filename)
    # open the file, read it, parse it,
    module = parse_file(filename)

    # extract exported declarations, produce a global directive for them and
    # put them into context

    # Collect declarations
    declarations = list(
        filter(lambda x: x.type == parser.NodeType.DECLARATION, module.children)
    )
    for dec in declarations:
        extern = dec.children[0].token and dec.children[0].token.typ == TokenSpec.EXTERN
        var = typed_identifier_to_variable(
            dec.children[0] if not extern else dec.children[1]
        )
        var.location = AddressPointer(var.identifier)
        context.globals_[var.identifier] = var
        asm.append(f"extern {var.identifier}")

    # Collect functions
    functions = collect_functions_from_module(module, context)
    for function in functions:
        asm.append(f"extern {function.identifier} ")

    return asm, []


def collect_children_of_type(parent, type: NodeType):
    nodes = [parent]
    collection = []
    while len(nodes) > 0:
        n = nodes.pop(0)
        for child in n.children:
            nodes.append(child)
        if n.type == type:
            collection.append(n)
    return collection


@node_compiler(parser.NodeType.MODULE)
def compile_module(node, context: Context) -> list[str]:
    asm = []
    # Collect imports
    imports = list(filter(lambda x: x.type == parser.NodeType.IMPORT, node.children))
    for i in imports:
        code, _ = compile(i, context)
        asm.extend(code)

    # Store literals into .rodata
    asm.append("section .rodata")
    # Collect all strings together to keep in .ro
    string_literals = collect_children_of_type(node, NodeType.STRING)
    for n in string_literals:
        s_id = f"___s{len(context.string_literals) + 1}"
        context.string_literals[n.token.text] = s_id
        asm.append(f"{s_id}: db  {string_to_nasm(n.token.text)}, 0")

    # Literal floats have to be stored
    float_literals = collect_children_of_type(node, NodeType.FLOAT)
    for flt_node in float_literals:
        f_id = f"___f{len(context.float_literals) + 1}"
        asm.append(f"{f_id}: dq {flt_node.token.text}")
        var = Variable(
            identifier=f_id,
            type=VariableType.f64,
            indirection_count=0,
            location=f"[{f_id}]",
        )
        context.float_literals[flt_node.token.text] = var

    # Collect declarations
    declarations = list(
        filter(lambda x: x.type == parser.NodeType.DECLARATION, node.children)
    )
    asm.append("section .data")
    for declaration in filter(lambda d: len(d.children) > 1, declarations):
        code, _ = compile(declaration, context)
        asm.extend(code)
    asm.append("")
    asm.append("section .bss")
    for declaration in filter(lambda d: len(d.children) < 2, declarations):
        code, _ = compile(declaration, context)
        asm.extend(code)
    asm.append(" ")

    # Collect funcs, adds them to context
    functions = collect_functions_from_module(node, context)

    # Compile definitions of the collected functions
    asm.append("section .text")
    for function in functions:
        code, _ = compile(function.node, context)
        asm.extend(code)

    assert len(context.occupied_registers) == 0, (
        f"Taken regs: {context.occupied_registers}\n{'\n'.join(asm)}'"
    )
    return asm, []


def string_to_nasm(s):
    n = [""]
    p = 0
    escaping = False
    while p < len(s):
        c = s[p]
        p += 1
        if escaping and c == "\\":
            n[-1].append("\\")
            escaping = False
        elif escaping and c == "n":
            n.append(ord("\n"))
            n.append("")
            escaping = False
        elif escaping and c == "t":
            n.append(ord("\t"))
            n.append("")
            escaping = False
        elif escaping and c == "0":
            n.append(ord("\0"))
            n.append("")
            escaping = False
        elif c == "\\":
            escaping = True
            continue
        else:
            n[-1] += c
    return ", ".join(map(lambda x: f'"{x}"' if isinstance(x, str) else f"{x}", n))


def test_str_nasm():
    print(string_to_nasm("this\\0that\\nIS\\tagain"))


# def load_var_to_register(context, var):
#     if var.indirection_count or not var.type == VariableType.f64:
#         reg = context.take_a_register()
#         reg.type = var.type
#         reg.indirection_count = var.indirection_count
#         code = [
#             f"mov {reg}, {var}",
#         ]
#     else:
#         reg = context.take_a_float_register()
#         code = [
#             f"mov {reg}, {var}",
#         ]
#     return reg, code


@node_compiler(parser.NodeType.DECLARATION)
def compile_declaration(node, context: Context):
    extern = node.children[0].token and node.children[0].token.typ == TokenSpec.EXTERN

    var = typed_identifier_to_variable(
        node.children[0] if not extern else node.children[1]
    )
    init = None
    if len(node.children) > (2 if extern else 1):
        if extern:
            raise CompileError("Can't set a intialiser to an external. ", node)
        init = node.children[1]

    if extern or not context.use_stack:

        if context.is_local_var(var.identifier):
            raise CompileError(
                f"Variable '{var.identifier}' is already defined.", node.children[0]
            )

        var.location = AddressPointer(var.identifier)
        context.globals_[var.identifier] = var

        if extern:
            return [
                f"extern {var.identifier}"
            ], []  # todo this is going in the data sect

        if init:
            code, (literal,) = compile(init, context)
            if not isinstance(literal, Literal):
                raise CompileError(
                    f"Can't initialise '{var.identifier}' with non-constant value.",
                    init,
                )
            # todo check sizes match
            dd = SizeDefiners[var.size]
            return [
                f"global {var.identifier}",
                f"{var.identifier}: {dd} {literal.value}",
            ], []

        res = SizeReserves[var.size]
        count = 1 if not isinstance(var, Array) else var.length
        return [f"global {var.identifier}", f"{var.identifier}: {res} {count}"], []
    else:
        if context.is_local_var(var.identifier):
            raise CompileError(
                f"Variable '{var.identifier}' is already defined.", node.children[0]
            )

        count = 1 if not isinstance(var, Array) else var.length
        context.locals_stack_size += count * var.size
        var.location = RBP(-context.locals_stack_size)
        context.variables[var.identifier] = var
        initialise = ()
        if init:
            if init.type == NodeType.STRING:
                if var.type != VariableType.u8 and not isinstance(var, Array):
                    raise CompileError("Only asign strings to u8[] types.", node)
                src_str = init.token.text
                src_id = context.string_literals.get(src_str)
                if not src_id:
                    raise CompileError(
                        f'Compiler bug. Missing string literal "{src_str}" in ro data.',
                        node,
                    )
                str_len = len(src_str)
                if str_len > var.length:
                    raise CompileError(
                        f"'{src_str} doesnt fit into {var.identifier}. Needs {str_len} bytes.",
                        node,
                    )
                initialise = [
                    "cld",
                    f"mov rsi, {src_id}         ; source address",
                    f"lea rdi, {var.location}   ; destination address",
                    f"mov rcx, {str_len}                 ; number of bytes to copy",
                    "rep movsb",
                ]
            else:
                sizespec = SizeSpecifiers[var.size]
                asm, (reg,) = compile(init, context)
                if not isinstance(reg, Literal):  # err hack ot avoid a:8 = u8|9|
                    reg = ensure_is_a_register(asm, reg, context)
                if var.type != reg.type:
                    if isinstance(reg, Literal):
                        reg.type = var.type
                    else:
                        raise CompileError(
                            f"Type mismatch, {var.identifier} is {var.type} while rval({reg}) is {reg.type}",
                            node,
                        )
                initialise = [*asm, f"mov {sizespec} {var.location}, {reg}"]
                context.mark_free_if_reg(reg)

        return [f"; {var.identifier} @ {var.location}", *initialise], []


@node_compiler(NodeType.FUNC_DECL)
def compile_func_decl(node, context: Context):
    identifier = node.children[0]
    return [f"extern {identifier.token.text}"], []


@node_compiler(NodeType.FUNC_DEF)
def compile_func(node, context: Context):
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
            location = RBP(-rbp_offset)
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

    blk, _ = compile(block, context)
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
    ], []


@node_compiler(NodeType.BLOCK)
def compile_block(node, context: Context):
    asm = []
    context.new_frame()
    for child in node.children:
        print(child.type)
        code, registers = compile(child, context)
        asm.extend(code)
        for reg in registers:
            # some statements such a funccall may have taken a register
            context.mark_free_if_reg(reg)
    context.pop_frame()
    return asm, []


@node_compiler(NodeType.IDENTIFIER_REF)
def compile_ref(node, context: Context):
    var_name = node.children[0].token.text
    var = context.variables.get(var_name)
    if not var:
        raise CompileError("Unknown variable '{var_name}'", node.children[0])

    reg = context.take_a_register()
    reg.type = var.type
    reg.indirection_count = var.indirection_count + 1
    asm = [f"lea {reg}, {var.location}"]
    return asm, [reg]


def ensure_is_a_register(code, maybe_reg, context):
    """Ensures value is in a register by allocating one if needed.

    :param code: List of assembly instructions to append new instructions to
    :param maybe_reg: Register or other value that needs to be in a register
    :param context: Compilation context for register allocation
    :return: Tuple of (assembly instructions, register containing value)
    """
    if isinstance(maybe_reg, Register):
        return maybe_reg
    else:
        reg = context.take_a_register()
        reg.type = maybe_reg.type
        reg.indirection_count = maybe_reg.indirection_count
        code.append(f"mov {reg.full_reg}, {maybe_reg}")
        return reg


@node_compiler(NodeType.DE_REF)
def compile_de_ref(node, context: Context):
    # note, this is compiling only a an rval, in assignment this is not run
    code, (ptr,) = compile(node.children[0], context)
    reg = ensure_is_a_register(code, ptr, context)

    if reg.indirection_count < 1:
        raise CompileError("Problem. Can't dereference a non-reference.", node)
    reg.indirection_count -= 1
    asm = [*code]  # f"mov qword {reg}, {reg.location}"]
    # use the same register, drop size
    # can't use the same register as need to zero upper bits..
    asm.append(
        f"mov {SizeSpecifiers[reg.size]}  {reg}, [{reg.full_reg}]; ;cnt={reg.indirection_count}"
    )
    return asm, [reg]


@node_compiler(NodeType.ARRAY_ITEM)
def compile_array_item(node, context: Context):
    # note, this is compiling only as an rval, in assignment this is not run
    identifier_node, index_expr = node.children
    code, (index,) = compile(index_expr, context)
    identifier = context.variables[identifier_node.token.text]

    reg = context.take_a_register()
    tmp = context.free_registers[0]
    tmp.type = index.type
    tmp.indirection_count = 0
    reg.type = identifier.type
    reg.indirection_count = index.indirection_count
    code = [
        *code,
        f"lea {reg.full_reg}, {identifier}",
        f"xor {tmp.full_reg}, {tmp.full_reg}",  # horid temp to zero upper bits for addition
        f"mov {tmp}, {index}",
        *((f"add {reg.full_reg}, {tmp.full_reg}",) * identifier.size),
    ]

    asm = [*code]
    # use the same register, drop size
    asm.append(f"mov {SizeSpecifiers[reg.size]}  {reg}, [{reg.full_reg}] ; load it ")
    context.mark_free_if_reg(index)
    return asm, [reg]


@node_compiler(NodeType.IDENTIFIER)
def compile_identifier(node, context: Context):
    var_name = node.token.text
    var = context.variables.get(var_name)
    if not var:
        raise CompileError("Unknown variable '{var_name}'", node)
    return [], [context.variables[var_name]]


@node_compiler(NodeType.INTEGER)
def compile_integer(node, context: Context):
    # todo literal sizes and types
    return [], [Literal(node.token.text, type=VariableType.u32)]


@node_compiler(NodeType.CHARACTER)
def compile_literal_character(node, context: Context):
    return [], [Literal(str(ord(node.token.text)), type=VariableType.u8)]


@node_compiler(NodeType.FLOAT)
def compile_literal_float(node, context: Context):
    return [], [context.float_literals[node.token.text]]


@node_compiler(NodeType.STRING)
def compile_literal_string(node, context: Context):
    val = string_to_nasm(node.token.text)
    return [], [Literal(val, type=VariableType.u8, length=len(node.token.text))]


@node_compiler(NodeType.ADDITIVE)
def compile_additive(node: Node, context: Context):
    op = "add" if node.token == TokenSpec.PLUS else "sub"
    code_l, (l,) = compile(node.children[0], context)

    reg = ensure_is_a_register(code_l, l, context)

    code_r, (r,) = compile(node.children[1], context)
    if not isinstance(r, Literal) and r.size != reg.size:
        raise CompileError(
            f"Size mismatch. {r} is {r.size} bytes, {l} is {l.size} bytes. May need to squelch.",
            node,
        )
    if r.type != reg.type or (r.indirection_count > 1 != reg.indirection_count > 1):
        print(
            "WARNING: Type mismatch in addition. Sizes match so proceeding. May be bad."
        )

    context.mark_free_if_reg(r)

    ref_muliplier = 1
    if reg.indirection_count:
        reg.indirection_count -= 1
        ref_muliplier = reg.size
        reg.indirection_count += 1

    return [
        *code_l,
        *code_r,  #
        *(f"{op} {reg}, {r}" for _ in range(ref_muliplier)),
    ], [reg]


@node_compiler(NodeType.TERM)
def compile_term(node: Node, context: Context):
    code_l, (l,) = compile(node.children[0], context)

    reg = ensure_is_a_register(code_l, l, context)

    code_r, (r,) = compile(node.children[1], context)
    if not isinstance(r, Literal) and r.type != reg.type:
        raise CompileError(
            f"Size mismatch. {r} is {r.size} bytes {r.type}, {l} is {l.size} bytes {l.type}. May need to squelch.",
            node,
        )
    if r.type != reg.type or (r.indirection_count > 1 != reg.indirection_count > 1):
        print(
            "WARNING: Type mismatch in addition. Sizes match so proceeding. May be bad."
        )

    context.mark_free_if_reg(r)

    if r.indirection_count or reg.indirection_count:  # either is indirection
        raise CompileError("Can't multiply or divide with a pointer.", node)

    asm = [
        *code_l,
        *code_r,
    ]

    # if r is not a reg, make it so because mul needs registers
    if isinstance(r, Literal):
        rcx = Register("rcx")
        rcx.type = r.type
        asm.append(f"mov {rcx}, {r}")
        r = rcx

    rax = Register("rax")
    rax.type = l.type
    rdx = Register("rdx")
    rdx.type = l.type
    is_signed = reg.type in [
        VariableType.i8,
        VariableType.i16,
        VariableType.i32,
        VariableType.i64,
    ]
    signed_extend = {8: "cqo", 4: "cdq", 2: "cwd", 1: "cbw"}
    match node.token:
        case TokenSpec.ASTRISK:
            return [
                *asm,  #
                f"mov {rax}, {l}",
                f"{'mul' if not is_signed else 'imul'} {r.sizespec} {r}",
                f"mov {reg}, {rax}",
            ], [reg]
        case TokenSpec.DIVIDE:
            return [
                *asm,  #
                f"mov {rax}, {l}",
                "xor rdx, rdx" if not is_signed else signed_extend[rax.size],
                f"{'div' if not is_signed else 'idiv'} {r.sizespec} {r}",
                f"mov {reg}, {rax}",
            ], [reg]

        case TokenSpec.MODULO:
            return [
                *asm,  #
                f"mov {rax}, {l}",
                "xor rdx, rdx" if not is_signed else signed_extend[rax.size],
                f"{'div' if not is_signed else 'idiv'} {r.sizespec} {r}",
                f"mov {reg}, {rdx}",
            ], [reg]
        case _:
            raise CompileError("Compiler bug.", node)


@node_compiler(NodeType.ASSIGNMENT)
def compile_assignment(node, context: Context):
    # compile what will be assigned to it
    asm, (reg,) = compile(node.children[1], context)

    # put it into a register as some ops cant work off memory
    if not isinstance(
        reg, Literal
    ):  # this is a hack to allow a:u8=9 otherwise 9 is a u32 and needs squelching
        reg = ensure_is_a_register(asm, reg, context)

    if node.children[0].type == NodeType.DE_REF:
        decount = 1
        id_node = node.children[0].children[0]
        while id_node.type == NodeType.DE_REF:
            decount += 1
            id_node = id_node.children[0]
        get_addr_asm, (lval,) = compile(id_node, context)

        if lval.indirection_count < decount:
            raise CompileError("Tried to deref too many times in lval.", node)

        lval.indirection_count -= decount
        sizespec = SizeSpecifiers[lval.size]
        lval.indirection_count += decount

        if isinstance(lval, Register):
            target_reg = lval
            load_value = []
        else:
            # not a register, need to put it in one to be able to deref in x86
            target_reg = context.free_registers[0]
            load_value = [
                f"mov {target_reg.full_reg}, {lval.location}",
            ]

        # need to deref decount-1 more times now
        load_value.extend(
            (f"mov {target_reg.full_reg}, [{target_reg.full_reg}]",) * (decount - 1)
        )

        target_address = target_reg.full_reg

        if not isinstance(reg, Register):
            tmp = context.free_registers[1]
            tmp.type = lval.type
            tmp.indirection_count = 0
            load_value.append(f"mov {tmp.location}, {reg}  ; got to be in a reg")
            reg = tmp

        context.mark_free_if_reg(reg)
        context.mark_free_if_reg(target_reg)

        return [
            *asm,
            *get_addr_asm,
            *load_value,
            f"mov {sizespec} [{target_address}], {reg}",  # assign the value
        ], []

    elif (
        node.children[0].type == NodeType.ARRAY_ITEM
    ):  # todo I don't like this specialism branching
        identifier_node, index_expr = node.children[0].children
        code, (index,) = compile(index_expr, context)
        identifier = context.variables[identifier_node.token.text]
        sizespec = SizeSpecifiers[identifier.size]
        if isinstance(index, Register):  # its a register so use it
            target_address = index
        else:
            target_address = context.take_a_register()
            target_address.type = identifier.type
            target_address.indirection_count = identifier.indirection_count + 1
            tmp = context.free_registers[0]
            tmp.type = index.type
            tmp.indirection_count = 0
            code = [
                *asm,
                *code,
                f"lea {target_address.full_reg}, {identifier}",
                f"xor {tmp.full_reg}, {tmp.full_reg}",  # horid temp to zero upper bits for addition
                f"mov {tmp}, {index}",
                *(
                    (f"add {target_address.full_reg}, {tmp.full_reg}",)
                    * identifier.size
                ),
            ]

        asm = [
            *code,
            f"mov {sizespec} [{target_address}], {reg}   ; {identifier}[...]=... ",  # assign the value
        ]
        context.mark_free_if_reg(reg)
        context.mark_free_if_reg(target_address)
        return asm, []
    else:
        var = context.variables[node.children[0].token.text]
        sizespec = SizeSpecifiers[var.size]
        if var.size != reg.size:
            if isinstance(reg, Literal):
                reg.type = var.type
            else:
                raise CompileError("Size mismatch.", node)

        context.mark_free_if_reg(reg)
        return [*asm, f"mov {sizespec} {var.location}, {reg}"], []


@node_compiler(NodeType.RETURN)
def compile_return(node, context: Context):
    if len(node.children) == 1:
        ret = node.children[0]
        asm, (reg,) = compile(ret, context)
        rax = Register("rax")
        rax.type = reg.type
        rax.indirection_count = reg.indirection_count
        asm = (*asm, "xor rax, rax", f"mov {rax}, {reg}")
        context.mark_free_if_reg(reg)
    else:
        asm = []
    return [*asm, f"jmp {context.current_func.identifier}___ret"], []


@node_compiler(NodeType.TERMINATOR)
def compile_terminator(node, context: Context):
    return [], []


@node_compiler(NodeType.FUNC_CALL)
def compile_call(node, context: Context):
    function_name = node.children[0].token.text
    try:
        function = context.variables[function_name]
        assert isinstance(function, Function)
    except:
        raise CompileError(f"No function {function_name}.", node)

    param_fill_asm = []
    param_push_asm = []
    param_pop_asm = []
    input_params = node.children[1:]
    for parameter, i, parameter_decl in zip(
        input_params, range(len(function.parameters)), function.parameters
    ):
        target_register = ArgumentRegistersBySize[parameter_decl.size][i]
        target_register_full = ArgumentRegistersBySize[8][i]
        asm, (p_reg,) = compile(parameter, context)
        if (
            p_reg.type != parameter_decl.type
            or p_reg.indirection_count != parameter_decl.indirection_count
        ):
            raise CompileError(
                "Function call parameter type mismatch. "
                + f"'{parameter_decl.identifier}' must be {parameter_decl.type} {'ref' if parameter_decl.indirection_count else ''}",
                node,
            )
        asm.append(
            f"mov {SizeSpecifiers[parameter_decl.size]} {target_register}, {p_reg}"
        )
        param_push_asm.append(f"push {target_register_full}")
        param_pop_asm.append(f"pop {target_register_full}")
        param_fill_asm.extend(asm)
        context.mark_free_if_reg(p_reg)
    param_pop_asm = reversed(param_pop_asm)
    rax = Register("rax")
    rax.type = function.type
    rax.indirection_count = function.indirection_count
    return_val = context.take_a_register()
    return_val.type = function.type
    return_val.indirection_count = function.indirection_count

    return [
        *param_push_asm,
        *param_fill_asm,
        "xor rax, rax",
        f"call {function_name}",
        f"xor {return_val.full_reg}, {return_val.full_reg}",
        f"mov {return_val}, {rax}",
        *param_pop_asm,
    ], [return_val]


@node_compiler(NodeType.FOR_ARRAY_LOOP)
def compile_for(node: Node, context: Context):
    array_id, index_id, element, block = node.children
    loop_id = f"loop_{node.id}"
    array = context.variables.get(array_id.token.text)
    if not array:
        raise CompileError(f"Unknown variable {array_id.token.text}", node)
    if not isinstance(array, Array):
        raise CompileError(
            f"{array_id} is not an array. For loops are for arrays.", node
        )
    context.new_frame()

    # index variable creation
    index = Variable(identifier=index_id.token.text, type=VariableType.u64)
    context.locals_stack_size += index.size
    index.location = RBP(-context.locals_stack_size)
    context.variables[index.identifier] = index

    element = Variable(
        identifier=element.token.text, type=array.type, indirection_count=1
    )
    context.locals_stack_size += element.size
    element.location = RBP(-context.locals_stack_size)
    context.variables[element.identifier] = element

    context.break_out_point.append(f".{loop_id}_end")
    context.continue_point.append(f".{loop_id}_tail")
    loop_body, _ = compile(block, context)
    context.break_out_point.pop()
    context.continue_point.pop()

    asm = [
        f".{loop_id}_begin:",
        f"mov qword {index.location}, 0        ; zero index",
        f"lea rax, {array.location}",
        f"  mov {element.location}, rax",
        f".{loop_id}_start:",
        f"  mov eax, {index.location}",
        f"  cmp eax, {array.length}",
        f"  jge .{loop_id}_end",
        *loop_body,
        f".{loop_id}_tail:",
        f"  add qword {element.location}, {element.base_type_size}",
        f"  add qword {index.location}, 1        ; inc index",
        f"   jmp .{loop_id}_start",
        f".{loop_id}_end:",
    ]

    context.pop_frame()
    return asm, []


@node_compiler(NodeType.WHILE_LOOP)
def compile_while(node: Node, context: Context):
    relational_node, block = node.children
    loop_id = f"while_{node.id}"

    condition_asm, (condition,) = compile(relational_node, context)

    context.break_out_point.append(f".{loop_id}_end")
    context.continue_point.append(f".{loop_id}_start")
    loop_body, _ = compile(block, context)
    context.break_out_point.pop()
    context.continue_point.pop()
    reg_for_comparer = Register("rax")
    reg_for_comparer.type = condition.type
    asm = [
        f".{loop_id}_start:",
        *condition_asm,
        "; check for true",
        f"mov {reg_for_comparer}, {condition.location}",
        f"cmp byte {reg_for_comparer}, 1",
        f"jnz .{loop_id}_end",
        *loop_body,
        f"jmp .{loop_id}_start",
        f".{loop_id}_end:",
    ]
    context.mark_free_if_reg(condition)

    return asm, []


@node_compiler(NodeType.BOOLEAN)
def compile_for_bool(node: Node, context: Context):
    reg = context.take_a_register()
    reg.type = VariableType.u8
    reg.indirection_count = 0

    if node.token.typ == TokenSpec.TRUE:
        asm = [f"mov {reg}, 1"]
    else:
        asm = [f"mov {reg}, 0"]
    return asm, [reg]


@node_compiler(NodeType.FOR_RANGE_LOOP)
def compile_for_range(node: Node, context: Context):
    range_node, identifier_node, block = node.children
    loop_id = f"loop_{node.id}"
    var = typed_identifier_to_variable(identifier_node)
    # todo check the var is an integer...

    # context.new_frame()
    context.locals_stack_size += var.size
    var.location = RBP(-context.locals_stack_size)
    context.variables[var.identifier] = var

    start_asm, (start,) = compile(range_node.children[0], context)
    end_asm, (end,) = compile(range_node.children[1], context)
    step_asm, (step,) = (
        compile(range_node.children[2], context)
        if len(range_node.children) > 2
        else ([], (1,))
    )
    context.break_out_point.append(f".{loop_id}_end")
    context.continue_point.append(f".{loop_id}_tail")
    loop_body, _ = compile(block, context)
    context.break_out_point.pop()
    context.continue_point.pop()
    reg_for_comparer = Register("rax")
    reg_for_comparer.type = var.type
    asm = [
        *start_asm,
        *end_asm,
        *step_asm,
        f".{loop_id}_begin:",
        f"mov {var.sizespec} {var.location}, {start}        ; range start",
        f".{loop_id}_start:",
        f"  mov {reg_for_comparer}, {var.location}",
        f"  cmp {reg_for_comparer}, {end}",
        f"  jge .{loop_id}_end",
        *loop_body,
        f".{loop_id}_tail:",
        f"  add {var.sizespec} {var.location}, {step}        ; inc index",
        f"   jmp .{loop_id}_start",
        f".{loop_id}_end:",
    ]
    context.mark_free_if_reg(start)
    context.mark_free_if_reg(end)
    context.mark_free_if_reg(step)
    # context.pop_frame()
    return asm, []


@node_compiler(NodeType.RELATIONAL)
def compile_relational(node, context: Context):
    left, right = node.children
    left_asm, (left,) = compile(left, context)
    right_asm, (right,) = compile(right, context)

    if left.type in SignedIntVariableTypes or right.type in SignedIntVariableTypes:
        is_signed = True
    else:
        is_signed = False

    condition = node.token.typ
    match condition:
        case TokenSpec.LESS_THAN:
            op = "setb" if not is_signed else "setl"
        case TokenSpec.MORE_THAN:
            op = "seta" if not is_signed else "setg"
        case TokenSpec.LESS_THAN_EQ:
            op = "setbe" if not is_signed else "setle"
        case TokenSpec.MORE_THAN_EQ:
            op = "setae" if not is_signed else "setge"
        case TokenSpec.EQUAL:
            op = "sete"
        case TokenSpec.NOT_EQ:
            op = "setne"

    result = context.take_a_register()
    result.type = VariableType.u8
    result.indirection_count = 0
    rax = Register("rax")
    rax.type = left.type

    asm = [
        *left_asm,
        *right_asm,
        f"mov {rax}, {left}",
        f"cmp {rax}, {right}",  #
        f"{op} al",
        f"mov {result}, al",
    ]
    context.mark_free_if_reg(left)
    context.mark_free_if_reg(right)
    return asm, [result]


@node_compiler(NodeType.NEGATED_RELATIONAL)
def compile_negated_relational(node, context: Context):
    rel_asm, (relational,) = compile(node.children[0], context)

    if isinstance(relational, Register):
        target_reg = relational
        load_value = []
    else:
        # not a register, need to put it in one to be able to xor it safe
        target_reg = context.take_a_register()
        target_reg.type = relational.type
        target_reg.indirection_count = relational.indirection_count
        load_value = [
            f"mov {target_reg}, {relational.location}",
        ]
        context.mark_free_if_reg(relational)

    return [*rel_asm, *load_value, f"xor {target_reg.sizespec} {target_reg}, 1"], [
        target_reg
    ]


@node_compiler(NodeType.BREAK)
def compile_break(node, context: Context):
    if not context.break_out_point:
        raise CompileError("Can't break outside of a loop.", node)
    jmp_to = context.break_out_point[-1]  # dont pop it as it gets popped by the loop
    return [f"jmp {jmp_to}"], []


@node_compiler(NodeType.CONTINUE)
def compile_break(node, context: Context):
    if not context.continue_point:
        raise CompileError("Can't continue outside of a loop.", node)
    jmp_to = context.continue_point[-1]  # dont pop it as it gets popped by the loop
    return [f"jmp {jmp_to}"], []


@node_compiler(NodeType.SQUELCH)
def compile_squelch(node, context: Context):
    target_type, var_node = node.children
    target_type = TokenVariableTypeMap[target_type.token.typ]
    target_size = TypeSizes[target_type]

    prep_var_asm, (var,) = compile(var_node, context)
    if isinstance(var, Literal):
        var.type = target_type
        return [], [var]

    reg = context.take_a_register()
    reg.type = target_type
    reg.indirection_count = var.indirection_count

    source_size = var.size
    if target_size > source_size:
        # squelch up
        reg.type = var.type
    else:
        # squelch down
        reg.type = target_type

    if isinstance(var, Register):
        # could be moving a reg to a reg so need to match
        var.type = reg.type

    asm = [
        *prep_var_asm,
        f"xor {reg.full_reg}, {reg.full_reg}",  # zero it
        f"mov {reg}, {var};",
    ]
    reg.type = target_type
    context.mark_free_if_reg(var)
    return asm, [reg]


@node_compiler(NodeType.IF_STATEMENT)
def compile_if(node, context: Context):
    condition = node.children[0]
    eval_expr, (condition,) = compile(condition, context)

    if isinstance(condition, Literal):
        c = context.free_registers[0]
        c.type = condition.type
        c.indirection_count = condition.indirection_count
        comparison = [f"mov {c}, {condition}", f"cmp byte {c}, 0"]
    else:
        comparison = [
            f"cmp byte {condition}, 0",
        ]

    pos_label = f".pos_{id(node)}"
    end_label = f".end_if_{id(node)}"
    comparison = [
        *comparison,
        f"jnz {pos_label}",
    ]
    context.mark_free_if_reg(condition)

    negative_block, _ = (
        compile_block(node.children[2], context) if len(node.children) > 2 else ([], [])
    )

    positive_block, _ = compile_block(node.children[1], context)

    return [
        *eval_expr,
        *comparison,
        *negative_block,
        f"jmp {end_label}",
        f"{pos_label}:",
        *positive_block,
        f"{end_label}:",
    ], []


def parse_file(file, debug=False):
    with open(file, "r") as in_f:
        src = in_f.read()
    tokens = list(TokenSpec.tokenise(src))
    if debug:
        for p in tokens:
            print(p.typ, p.text)
    module = Module(tokens)
    return module


def compiler(file, debug=False):
    module = parse_file(file, debug)

    if not module:
        return None
    print_tree(module)
    return compile(module, Context(filename=file))[0]
