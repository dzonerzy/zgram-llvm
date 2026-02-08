# zgram-llvm

Minimal repackaged LLVM static libraries for [zgram](https://github.com/dzonerzy/zgram)'s JIT compilation pipeline.

Contains only the static libs and C API headers needed for ORC JIT + X86 backend + optimizations, extracted from official LLVM releases.

## Platforms

| Platform | Tarball |
|---|---|
| Linux x86_64 | `llvm-{version}-x86_64-linux.tar.xz` |
| Linux aarch64 | `llvm-{version}-aarch64-linux.tar.xz` |
| Windows x86_64 | `llvm-{version}-x86_64-windows.tar.xz` |
| Windows aarch64 | `llvm-{version}-aarch64-windows.tar.xz` |
| macOS ARM64 | `llvm-{version}-aarch64-macos.tar.xz` |

## Contents

Each tarball contains:
- 61 LLVM static libraries (Core, JIT, X86 backend, CodeGen, optimizations, etc.)
- 30 LLVM C API headers (`llvm-c/*.h`)
- libc++, libc++abi, libunwind static libs (Linux/macOS only)

## How it works

A CI workflow runs on every push to `main`. It downloads the official LLVM release tarballs, extracts only the needed files, and uploads repackaged tarballs as a GitHub release.

To bump the LLVM version, update `LLVM_VERSION` in `repackage.py` and push.

## Usage

These tarballs are referenced as Zig `build.zig.zon` dependencies in zgram. They are automatically downloaded at build time.

## Local repackaging

```
python3 repackage.py                  # all platforms
python3 repackage.py x86_64-linux     # single platform
python3 repackage.py --list           # list platforms
```
