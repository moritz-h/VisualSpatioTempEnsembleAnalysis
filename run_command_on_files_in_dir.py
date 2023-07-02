# compile all files in provided directory and output to specified directoy
import argparse
import os
import re
import sys
import subprocess


def execute_command(cmd: str, src_dir: str, regex: str):
    files = os.listdir(src_dir)

    if regex is not None:
        pattern = re.compile(regex)

        files = [f for f in files if pattern.match(f)]

    commands = [cmd.format(f=f.rsplit('.', 1)[0], t=f.rsplit('.', 1)[-1]) for f in files]

    for command in commands:
        print("running command: '{}'".format(command))
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   env=os.environ.copy())
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stdout, stderr)


def main(args):
    parser = argparse.ArgumentParser("Runs command provided command (--cmd COMMAND) on all files f_i in provided source"
                                     " dir (--src_dir SOURCE_DIR). The provided command should be a format string which "
                                     "can access individual files with {f} (without filetype) and its file type with "
                                     "{t}.")

    parser.add_argument("--cmd", type=str, required=True,
                        help="The COMMAND format string. Access individual files with {f} without filetype and its "
                             "filetype with {t}.")
    parser.add_argument("--src_dir", type=str, required=True,
                        help="The source dir which contains the files to run the command on.")
    parser.add_argument("--regex", type=str, required=False,
                        help="A regular expression that specifies which files to pick in the source dir.")

    arguments = parser.parse_args(args[1:])
    execute_command(**vars(arguments))


if __name__ == '__main__':
    main(sys.argv)

