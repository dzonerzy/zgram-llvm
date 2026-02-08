# zgram-llvm

Pre-built LLVM static libraries for zgram's JIT compilation pipeline.

Contains only the minimal set of LLVM static libs and C API headers needed for ORC JIT compilation on supported platforms.

## Platforms

- Linux x86_64
- Windows x86_64
- Linux aarch64 (planned)

## Usage

These tarballs are referenced as Zig build dependencies in zgram's \.
They are automatically downloaded at build time.
