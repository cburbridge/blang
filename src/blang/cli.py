import sys
import click
import subprocess
from blang.compiler.compiler import compiler, CompileError
from blang.parser import ParseError
import tempfile
from pathlib import Path


@click.command
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--debug", is_flag=True, required=False, default=False)
def cli_cmd(input_files, output_file, debug):
    cli(input_files, output_file, debug)


def cli(input_files, output_file, debug=False):
    """Compiles INPUT_FILES and writes the linked binary to OUTPUT_FILE."""
    print(f"inp: {input_files}")
    object_files = []

    entry = [
        "global _start",
        "extern main",
        "; --- ENTRY POINT ---",
        "_start:",
        "  call main",
        "  mov edi, eax",
        "  mov eax, 60",
        "  syscall",
    ]
    with open("/tmp/entry.asm", "w") as tmp:
        #    with tempfile.NamedTemporaryFile("w") as tmp:
        tmp.write("\n".join(entry))
        tmp.flush()
        cmd = [
            "nasm",
            "-f",
            "elf64",
            tmp.name,
            "-o",
            str(Path(tmp.name).with_suffix(".o")),
        ]
        click.echo(">  " + " ".join(cmd))
        subprocess.check_call(cmd)
        object_files.append(Path(tmp.name).with_suffix(".o"))

    for file in input_files:
        with open(file, "r") as in_f:
            src = in_f.read()
        with tempfile.NamedTemporaryFile("w") as tmp:
            click.echo(f"Compiling {file}")
            try:
                asm = compiler(src)
            except CompileError as e:
                click.echo(f"Compilation error:\n {str(e)}", err=True)
                return
            except ParseError as e:
                click.echo(f"Syntax error:\n {str(e)}", err=True)
                return
            tmp.write("\n".join(asm or []))
            tmp.flush()
            if debug:
                print("******" * 5)
                print({file})
                for i, l in enumerate(asm or []):
                    print(f"{i + 1}\t{l}")
            cmd = ["nasm", "-f", "elf64", tmp.name, "-o", tmp.name + ".o"]
            click.echo(">" + " ".join(cmd))
            subprocess.check_call(cmd)
            object_files.append(tmp.name + ".o")

    click.echo(f"Linking {object_files}")
    cmd = ["ld", *map(str, object_files), "-o", output_file]
    click.echo("> " + " ".join(cmd))
    subprocess.check_call(cmd)
    click.echo(f"Executable written to {output_file}")


def main():
    try:
        cli_cmd()
    except subprocess.CalledProcessError:
        click.echo("OhNo.", err=True)
        return 1
