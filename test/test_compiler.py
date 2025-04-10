import os
from blang.cli import cli
import tempfile
import subprocess
from pathlib import Path
import pytest


def test_compile_dev():
    program = """
    glob: u8
    g: u8[4]
# a cool program
def add(p:u64, q:u64) : u64 {
    return p + q
}
    
def main(): u64 {
    glob=9
    v: u64 = 5
    q: u64 = 7
    s: u8[32];
    # <s + 5> = 9;
    ss: u8;
    s[27] = 2
    ss = s[27]
    s[0]=99;
    t: u64 = add(v,q);  # finally, comments ahoy
    
    return s[0]+s[27]
} 
    """

    binary = None
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(program)
        f.flush()
        binary = f.name + ".bin"
        cli([f.name], binary, debug=True)

    result = subprocess.call([binary])
    assert result == 101

    os.remove(binary)


TEST_PROGRAM_FILES = list(Path(__file__).parent.glob("test_*.bl"))
TEST_PROGRAM_NAMES = list(
    map(lambda x: str(x.name).replace(".", "_"), TEST_PROGRAM_FILES)
)


@pytest.mark.parametrize("test_program", TEST_PROGRAM_FILES, ids=TEST_PROGRAM_NAMES)
def test_programs(test_program):
    print(f"Testing: {test_program.name}")
    with open(test_program, "r") as f:
        code = f.read()
        # Do something with the content
        expected_exit = int(
            code.splitlines()[0].split("expected_exit_code")[1].split("=")[1]
        )
        print(f"Expecting exit code {expected_exit}")

        with tempfile.NamedTemporaryFile("w") as f:
            binary = f.name + ".bin"
            cli([test_program.absolute()], binary, debug=True)

            result = subprocess.call([binary])
            os.remove(binary)

        assert result == expected_exit
