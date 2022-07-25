"""
This file provides a utility which is used by the workflows to determine the exact options that correspond to a given
compiler version. This significantly simplifies the handling of the matrix strategy in the `build.yml` workflow.
"""
import sys

# which specification do we want to get? cxx [C++ compiler executable], cc [C compiler executable],
# or install [apt packages to install]
which = sys.argv[1]
# which compiler to target. Possible are gcc-# and clang-# with # a supported version number.
compiler = sys.argv[2]

vendor, _, version = compiler.partition("-")

CXX = {"gcc": "g++", "clang": "clang++"}
CC = {"gcc": "gcc", "clang": "clang"}
INSTALL = {"gcc": ["gcc", "g++"], "clang": ["clang"]}

if which == "cxx":
    print(CXX[vendor] + "-" + version)
elif which == "cc":
    print(CC[vendor] + "-" + version)
elif which == "install":
    print(" ".join([package + "-" + version for package in INSTALL[vendor]]))
exit(0)
