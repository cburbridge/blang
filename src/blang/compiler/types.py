from collections import defaultdict, ChainMap
from dataclasses import dataclass, field
import enum
from blang.parser import TokenSpec, Node


class VariableType(enum.StrEnum):
    u8 = enum.auto()
    u16 = enum.auto()
    u32 = enum.auto()
    u64 = enum.auto()

    i8 = enum.auto()
    i16 = enum.auto()
    i32 = enum.auto()
    i64 = enum.auto()

    flt = enum.auto()


TokenVariableTypeMap = {
    TokenSpec.U8: VariableType.u8,
    TokenSpec.U16: VariableType.u16,
    TokenSpec.U32: VariableType.u32,
    TokenSpec.U64: VariableType.u64,
    TokenSpec.I8: VariableType.i8,
    TokenSpec.I16: VariableType.i16,
    TokenSpec.I32: VariableType.i32,
    TokenSpec.I64: VariableType.i64,
    TokenSpec.FLOAT: VariableType.flt,
}

TypeSizes = {
    VariableType.u8: 1,
    VariableType.u16: 2,
    VariableType.u32: 4,
    VariableType.u64: 8,
    VariableType.i8: 1,
    VariableType.i16: 2,
    VariableType.i32: 4,
    VariableType.i64: 8,
    VariableType.flt: 8,
}

SizeReserves = {1: "resb", 2: "resw", 4: "resd", 8: "resq", 16: "resdq"}

SizeDefiners = {1: "db", 2: "dw", 4: "dd", 8: "dq", 16: "do"}
for t, s in TypeSizes.items():
    SizeDefiners[t] = SizeDefiners[s]

SizeSpecifiers = {1: "byte", 2: "word", 4: "dword", 8: "qword", 16: "oword"}
# for t, s in TypeSizes.items():
# SizeSpecifiers[t] = SizeSpecifiers[s]

ArgumentRegistersBySize = {
    8: ["rdi", "rsi", "rdx", "rcx", "r8", "r9"],
    4: ["edi", "esi", "edx", "ecx", "r8d", "r9d"],
    2: ["di", "si", "dx", "cx", "r8w", "r9w"],
    1: ["dil", "sil", "dxl", "cxl", "r8b", "r9b"],
}

# GP Registers split by size
RegistersPartial = {
    r: {1: f"{r}b", 2: f"{r}w", 4: f"{r}d", 8: f"{r}"}
    for r in ("r10", "r11", "r12", "r13", "r14", "r15")
}
RegistersPartial.update(
    {"rax": {8: "rax", 4: "eax", 2: "ax", 1: "al"}, "rbp": {8: "rbp"}}
)


class SizePropMixin:
    @property
    def size(self) -> int:
        return 8 if self.indirection_count else self.base_type_size

    @property
    def base_type_size(self) -> int:
        return TypeSizes[self.type]

    @property
    def sizespec(self):
        return SizeSpecifiers[self.size]


@dataclass
class Register(SizePropMixin):
    full_reg: str

    # Similar to a variable
    type: VariableType = None
    indirection_count: int = 0

    @property
    def content_type_size(self):
        return self.content_type and TypeSizes[self.content_type]

    @property
    def name(self):
        return RegistersPartial[self.full_reg][self.size]

    def partial(self, size):
        return RegistersPartial[self.full_reg][size]

    @property
    def location(self):
        return self.name

    def set_in_use(self, size):
        pass

    def __str__(self):
        return self.name or self.full_reg


@dataclass
class Literal(SizePropMixin):
    value: str

    # Similar to a variable
    type: VariableType = None
    indirection_count: int = 0

    def __str__(self):
        return self.value


@dataclass
class Variable(SizePropMixin):
    identifier: str
    type: VariableType

    location: str | None = None
    on_stack: bool = False
    indirection_count: int = 0
    external: bool = False
    exported: bool = False
    node: Node = None

    def __str__(self):
        return str(self.location)


@dataclass
class Array(SizePropMixin):
    identifier: str
    type: VariableType

    location: str | None = None
    on_stack: bool = False
    indirection_count: int = 0
    external: bool = False
    exported: bool = False
    node: Node = None
    length: int = 1  # Number of elements if an array

    def __str__(self):
        return str(self.location)


@dataclass
class AddressPointer:
    # in a register
    base: Register | str
    offset: int = 0

    def __str__(self):
        return f"[{self.address}]"

    @property
    def address(self):
        return f"{self.base}{f' + {self.offset}' if self.offset > 0 else f' - {abs(self.offset)}' if self.offset < 0 else ''}"


def test_ap():
    a = RBP(-9)
    print(f"{a}")
    a = RBP(9)
    print(f"{a}")
    a = RBP(0)
    print(f"{a}")
    assert False


def RBP(offset):
    return AddressPointer("rbp", offset)


@dataclass
class Function(Variable):
    parameters: list[Variable] = field(default_factory=list)


@dataclass
class Context:
    variable_stack: ChainMap = field(default_factory=ChainMap)
    locals_stack_size: int = 0
    use_stack: bool = False
    current_func: Function = None
    free_registers: list[Register] = field(
        default_factory=lambda: [
            Register("r10"),
            Register("r11"),
            Register("r12"),
            Register("r13"),
            Register("r14"),
            Register("r15"),
        ]
    )
    occupied_registers: list[Register] = field(default_factory=lambda: [])

    strings = {}

    @property
    def globals_(self):
        return self.variable_stack.maps[-1]

    @property
    def variables(self):
        return self.variable_stack

    def is_local_var(self, var):
        return var in self.variable_stack.maps[0]

    def pop_frame(self):
        self.variable_stack = self.variable_stack.parents

    def new_frame(self):
        self.variable_stack = self.variable_stack.new_child()

    def mark_free_if_reg(self, maybe_reg):
        if maybe_reg in self.occupied_registers:
            self.occupied_registers.remove(maybe_reg)
            self.free_registers.append(maybe_reg)


def test_var_stack():
    c = Context()
    c.globals_["a"] = 1
    c.new_frame()
    c._locals_["a"] = 5
