import os
from blang.cli import cli
import tempfile
import subprocess
from pathlib import Path
import pytest


TEST_PROGRAM_FILES = list(Path(__file__).parent.glob("test_*.bl"))
TEST_PROGRAM_NAMES = list(
    map(lambda x: str(x.name).replace(".", "_"), TEST_PROGRAM_FILES)
)


@pytest.mark.parametrize("test_program", TEST_PROGRAM_FILES, ids=TEST_PROGRAM_NAMES)
def test_programs(test_program):
    print(f"Testing: {test_program.name}")
    with open(test_program, "r") as f:
        code = f.read()
        code_lines = code.splitlines()
        # Do something with the content
        expected_exit = int(code_lines[0].split("expected_exit_code")[1].split("=")[1])
        print(f"Expecting exit code {expected_exit}")

        ld_flags = []
        for lin in code_lines:
            if "***ld_flag***" in lin:
                ld_flags.append(lin.split("***ld_flag***")[1])

        # extract expected output
        expected_output = []
        i = 1
        block = False
        for l in code_lines:
            if "***expected output***" in l:
                block = True
                start = False
            elif block and not start and "****" in l:
                indent = len(l.split(" ")[0]) + 1
                start = True
            elif block and start and "****" in l:
                block = False
            elif block and start:
                expected_output.append(l[indent:])

        print(f"Expecting stdout \n```\n{'\n'.join(expected_output)}\n```")

        with tempfile.NamedTemporaryFile("w") as f:
            binary = f.name + ".bin"
            try:
                if cli(test_program.absolute(), binary, " ".join(ld_flags), debug=True):
                    assert False, "compile failed"
            except Exception:
                raise
            process = subprocess.run([binary], capture_output=True, text=True)
            result = process.returncode
            output_lines = process.stdout.splitlines()

        assert result == expected_exit

        print(f"Program stdout:\n```\n{process.stdout}```")
        assert output_lines == expected_output, (
            f"Expected:\n{expected_output}\nGot:\n{output_lines}"
        )
        os.remove(binary)

