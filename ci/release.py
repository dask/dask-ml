"""
Start a release
"""
import argparse
import subprocess
import sys

from git import Repo
from packaging.version import parse, Version
import twine  # noqa


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version', help="Version to tag")
    parser.add_argument('-r', '--remote', default='upstream')
    return parser.parse_args(args)


def check(version):
    v = parse(version)
    assert isinstance(v, Version), f'Invalid version: {version}'


def main(args=None):
    args = parse_args(args)
    repo = Repo(".")
    remote = repo.remotes[args.remote]

    # TODO: verify that we're current with upstream/master
    # TODO: print the previous release
    # TODO: validate that the tag doesn't exist already
    # TODO: verify the version / tag naming scheme
    assert repo.active_branch.name == "master"

    check(args.version)
    print(f"Releasing for version {args.version}")
    while True:
        confirm = input(f"Confirm for {args.version} [y/n]: ")[0].lower()
        if confirm == 'y':
            break
        elif confirm == 'n':
            sys.exit(1)

    commit = repo.index.commit(f"RLS: {args.version}")
    tag = repo.create_tag(f"v{args.version}", message=f"RLS: {args.version}")

    print("Created commit: ", commit)
    print("Created tag   : ", tag)

    while True:
        confirm = input(f"Ready to push? [y/n]: ")[0].lower()
        if confirm == 'y':
            break
        elif confirm == 'n':
            sys.exit(1)

    remote.push("master:master")
    remote.push(tag)

    subprocess.check_call(["python", "setup.py", "sdist"])
    subprocess.check_call(["twine", "upload", "dist/*", "--skip-existing"])


if __name__ == '__main__':
    sys.exit(main(None))
