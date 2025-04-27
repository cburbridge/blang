import os
from blang.cli import cli
import tempfile
import subprocess
from pathlib import Path
import pytest

from blang.compiler.compiler import CompileError

TEST_PROGRAM_FILES = list(Path(__file__).parent.glob("test_*.bl"))
TEST_PROGRAM_NAMES = list(
    map(lambda x: str(x.name).replace(".", "_"), TEST_PROGRAM_FILES)
)


@pytest.mark.parametrize("test_program", TEST_PROGRAM_FILES, ids=TEST_PROGRAM_NAMES)
def test_programs(test_program):
    print(f"Testing: {test_program.name}")
    with open(test_program, "r") as f:
        code = f.read()
        code_lines =  code.splitlines()
        # Do something with the content
        expected_exit = int(
            code_lines[0].split("expected_exit_code")[1].split("=")[1]
        )
        ld_flags=[]
        i=1
        while "***ld_flag***" in code_lines[i]:
            ld_flags.append(code_lines[i].split("***ld_flag***")[1])
            i+=1
        print(f"Expecting exit code {expected_exit}")

        with tempfile.NamedTemporaryFile("w") as f:
            binary = f.name + ".bin"
            try:
                if cli(test_program.absolute(), binary," ".join(ld_flags), debug=True):
                    assert False, "compile failed"
            except Exception:
                raise
            result = subprocess.call([binary])
            os.remove(binary)

        assert result == expected_exit
