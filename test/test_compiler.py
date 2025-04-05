import os
from blang.cli import cli
import tempfile
import subprocess


def test_compile():
    program = """
# a cool program
def add(p:u64, q:u64) : u64 {
    return p + q
}
    
def main(): u64 {
    v: u64 = 5
    q: u64 = 7
    # this is nice
    t: u64 = add(v,q);  # finally, comments ahoy
    return t
} 
    """

    binary = None
    with tempfile.NamedTemporaryFile("w") as f:
        f.write(program)
        f.flush()
        binary = f.name + ".bin"
        cli([f.name], binary, debug=True)

    result = subprocess.call([binary])
    assert result == 12

    os.remove(binary)
