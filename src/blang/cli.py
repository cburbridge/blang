from subprocess import CalledProcessError

import click
import subprocess
from blang.compiler.compiler import (
    compiler,
    CompileError,
    parse_file,
    import_node_to_filepath,
)
from blang.parser import ParseError, NodeType
import tempfile
from pathlib import Path


@click.command
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--debug", is_flag=True, required=False, default=False)
@click.option("--ld_flags", is_flag=False, required=False, default="")
def cli_cmd(input_file, output_file, ld_flags, debug):
    cli(input_file, output_file, ld_flags, debug)


def cli(input_file, output_file, ld_flags="", debug=False):
    """Compiles INPUT_FILES and writes the linked binary to OUTPUT_FILE."""
    object_files = []
    input_files = [input_file]
    module = parse_file(input_file, debug)
    imports = list(filter(lambda x: x.type == NodeType.IMPORT, module.children))
    for import_node in imports:
        input_files.append(import_node_to_filepath(import_node, input_file))

    for asm in (Path(__file__).parent / "compiler").glob("*.asm"):
        cmd = [
            "nasm",
            "-f",
            "elf64",
            str(asm),
            "-o",
            str(asm.with_suffix(".o")),
        ]
        click.echo(">  " + " ".join(cmd))
        subprocess.check_call(cmd)
        object_files.append(str(asm.with_suffix(".o")))

    for file in input_files:
        with tempfile.NamedTemporaryFile("w") as tmp:
            click.echo(f"Compiling {file}")
            try:
                asm = compiler(file, debug)
            except CompileError as e:
                click.echo(f"Compilation error:\n {str(e)}", err=True)
                return 1
            except ParseError as e:
                click.echo(f"Syntax error:\n {str(e)}", err=True)
                return 1
            tmp.write("\n".join(asm or []))
            tmp.flush()
            if not asm:
                print("Problem compiling.")
                return 1
            if debug:
                print("******" * 5)
                print({file})
                for i, l in enumerate(asm or []):
                    print(f"{i + 1}\t{l}")
            cmd = ["nasm", "-f", "elf64", tmp.name, "-o", tmp.name + ".o"]
            click.echo(">" + " ".join(cmd))
            try:
                subprocess.check_call(cmd)
            except CalledProcessError as e:
                print("Problem assembling code.")
                print(e)
                return 1
            object_files.append(tmp.name + ".o")

    click.echo(f"Linking {object_files}")
    cmd = ["ld", *map(str, object_files), *(ld_flags.split()), "-o", output_file]
    click.echo("> " + " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except CalledProcessError as e:
        print("Problem linking code")
        print(e)
        return 1
    click.echo(f"Executable written to {output_file}")
    return 0


def main():
    try:
        cli_cmd()
    except subprocess.CalledProcessError:
        click.echo("OhNo.", err=True)
        return 1
