from blang.parser import NodeType, Node


_node_precompilers = {}


def node_precompiler(type: NodeType):
    def node_precompiler(method):
        def _wrap(node: Node):
            method(node)
            node.extra.precompiled = True

        _node_precompilers[type] = _wrap
        return _wrap

    return node_precompiler


def precompile(node):
    children = [node]
    while children:
        child = children.pop()
        children.extend(child.children)
        if child.type in _node_precompilers:
            _node_precompilers[child.type](child)
            child.extra.precompiled = True
        else:
            print(f"WARNING: No precompiler for {child.type}")


@node_precompiler(NodeType.MODULE)
def precompile_module(node): ...


@node_precompiler(NodeType.BASE_TYPE)
def precompile_base_type(node): ...


@node_precompiler(NodeType.REF_TYPE)
def precompile_ref_type(node): ...


@node_precompiler(NodeType.IDENTIFIER_REF)
def precompile_identifier_ref(node): ...


@node_precompiler(NodeType.ARRAY)
def precompile_array(node): ...


@node_precompiler(NodeType.IDENTIFIER)
def precompile_identifier(node): ...


@node_precompiler(NodeType.TYPED_IDENTIFER)
def precompile_typed_identifier(node): ...


@node_precompiler(NodeType.DECLARATION)
def precompile_declaration(node): ...


@node_precompiler(NodeType.ASSIGNMENT)
def precompile_assignment(node): ...


@node_precompiler(NodeType.INTEGER)
def precompile_integer(node): ...


@node_precompiler(NodeType.FLOAT)
def precompile_float(node): ...


@node_precompiler(NodeType.RETURN)
def precompile_return(node): ...


@node_precompiler(NodeType.FUNC_CALL)
def precompile_func_call(node): ...


@node_precompiler(NodeType.PARAMETER_LIST)
def precompile_parameter_list(node): ...


@node_precompiler(NodeType.FUNC_DEF)
def precompile_func_def(node): ...


@node_precompiler(NodeType.FUNC_DECL)
def precompile_func_decl(node): ...


@node_precompiler(NodeType.BLOCK)
def precompile_block(node): ...


@node_precompiler(NodeType.CAPTURED_EXPRESSION)
def precompile_captured_expression(node): ...


@node_precompiler(NodeType.DE_REF)
def precompile_de_ref(node): ...


@node_precompiler(NodeType.ARRAY_ITEM)
def precompile_array_item(node): ...


@node_precompiler(NodeType.BOOLEAN)
def precompile_boolean(node): ...


@node_precompiler(NodeType.ADDITIVE)
def precompile_additive(node): ...


@node_precompiler(NodeType.TERM)
def precompile_term(node): ...


@node_precompiler(NodeType.RELATIONAL)
def precompile_relational(node): ...


@node_precompiler(NodeType.LOGIC_AND)
def precompile_logic_and(node): ...


@node_precompiler(NodeType.LOGIC_OR)
def precompile_logic_or(node): ...


@node_precompiler(NodeType.NEGATED_RELATIONAL)
def precompile_negated_relational(node): ...


@node_precompiler(NodeType.IF_STATEMENT)
def precompile_if_statement(node): ...


@node_precompiler(NodeType.WHILE_LOOP)
def precompile_while_loop(node): ...


@node_precompiler(NodeType.BREAK)
def precompile_break(node): ...


@node_precompiler(NodeType.CONTINUE)
def precompile_continue(node): ...


@node_precompiler(NodeType.RANGE)
def precompile_range(node): ...


@node_precompiler(NodeType.FOR_ARRAY_LOOP)
def precompile_for_array_loop(node): ...


@node_precompiler(NodeType.FOR_RANGE_LOOP)
def precompile_for_range_loop(node): ...


@node_precompiler(NodeType.SQUELCH)
def precompile_squelch(node): ...


@node_precompiler(NodeType.STRING)
def precompile_string(node): ...


@node_precompiler(NodeType.CHARACTER)
def precompile_character(node): ...


@node_precompiler(NodeType.TERMINATOR)
def precompile_terminator(node): ...


@node_precompiler(NodeType.IMPORT)
def precompile_import(node): ...
