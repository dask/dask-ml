"""
Start a release
"""
import argparse
import subprocess
import sys

from git import Repo
from packaging.version import Version, parse


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Version to tag")
    parser.add_argument("-r", "--remote", default="upstream")
    parser.add_argument("--no-push", action="store_false")
    return parser.parse_args(args)


def check(version):
    v = parse(version)
    assert isinstance(v, Version), "Invalid version: {}".format(version)
    assert not version.startswith("v")


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
    print("Releasing for version {}".format(args.version))
    while True:
        confirm = input("Confirm for {} [y/n]: ".format(args.version))[0].lower()
        if confirm == "y":
            break
        elif confirm == "n":
            sys.exit(1)

    commit = repo.index.commit("RLS: {}".format(args.version))
    tag = repo.create_tag(
        "v{}".format(args.version), message="RLS: {}".format(args.version)
    )

    print("Created commit: ", commit)
    print("Created tag   : ", tag)

    if args.no_push:
        print("--no-push. Exiting")
        sys.exit(0)

    while True:
        confirm = input("Ready to push? [y/n]: ")[0].lower()
        if confirm == "y":
            break
        elif confirm == "n":
            sys.exit(1)

    remote.push("master:master")
    remote.push(tag)

    subprocess.check_call(["python", "setup.py", "sdist"])
    subprocess.check_call(["twine", "upload", "dist/*", "--skip-existing"])


if __name__ == "__main__":
    sys.exit(main(None))
