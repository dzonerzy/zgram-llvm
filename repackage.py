#!/usr/bin/env python3
"""
Repackage official LLVM release tarballs into minimal archives
containing only the static libs and C API headers needed for zgram's JIT pipeline.

Usage:
    python3 repackage.py                  # repackage all platforms
    python3 repackage.py x86_64-linux     # repackage single platform
    python3 repackage.py --list           # list available platforms

Output tarballs use standardized naming:
    llvm-{version}-{arch}-{os}.tar.xz
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

LLVM_VERSION = "21.1.8"

# Map from our standardized name -> official release asset name
PLATFORMS = {
    "x86_64-linux": f"LLVM-{LLVM_VERSION}-Linux-X64.tar.xz",
    "aarch64-linux": f"LLVM-{LLVM_VERSION}-Linux-ARM64.tar.xz",
    "x86_64-windows": f"clang+llvm-{LLVM_VERSION}-x86_64-pc-windows-msvc.tar.xz",
    "aarch64-windows": f"clang+llvm-{LLVM_VERSION}-aarch64-pc-windows-msvc.tar.xz",
    "aarch64-macos": f"LLVM-{LLVM_VERSION}-macOS-ARM64.tar.xz",
}

# The extracted top-level directory name differs per tarball
EXTRACT_DIRS = {
    "x86_64-linux": f"LLVM-{LLVM_VERSION}-Linux-X64",
    "aarch64-linux": f"LLVM-{LLVM_VERSION}-Linux-ARM64",
    "x86_64-windows": f"clang+llvm-{LLVM_VERSION}-x86_64-pc-windows-msvc",
    "aarch64-windows": f"clang+llvm-{LLVM_VERSION}-aarch64-pc-windows-msvc",
    "aarch64-macos": f"LLVM-{LLVM_VERSION}-macOS-ARM64",
}

RELEASE_URL = (
    f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}"
)

# Required LLVM static libs (without prefix/extension)
REQUIRED_LIBS = [
    # Core/IR
    "LLVMCore",
    "LLVMBitReader",
    "LLVMBitWriter",
    "LLVMIRReader",
    "LLVMIRPrinter",
    "LLVMAsmParser",
    "LLVMBitstreamReader",
    # JIT
    "LLVMOrcJIT",
    "LLVMJITLink",
    "LLVMExecutionEngine",
    "LLVMRuntimeDyld",
    "LLVMOrcTargetProcess",
    "LLVMOrcShared",
    # X86 backend
    "LLVMX86CodeGen",
    "LLVMX86Desc",
    "LLVMX86Info",
    "LLVMX86AsmParser",
    "LLVMX86Disassembler",
    "LLVMX86TargetMCA",
    # CodeGen / Optimization
    "LLVMCodeGen",
    "LLVMCodeGenTypes",
    "LLVMPasses",
    "LLVMCoroutines",
    "LLVMSelectionDAG",
    "LLVMGlobalISel",
    "LLVMScalarOpts",
    "LLVMInstCombine",
    "LLVMAggressiveInstCombine",
    "LLVMTransformUtils",
    "LLVMVectorize",
    "LLVMipo",
    "LLVMObjCARCOpts",
    "LLVMCFGuard",
    "LLVMInstrumentation",
    "LLVMLinker",
    # Target / MC
    "LLVMTarget",
    "LLVMTargetParser",
    "LLVMMC",
    "LLVMMCParser",
    "LLVMMCDisassembler",
    "LLVMAsmPrinter",
    "LLVMMCA",
    # Analysis / Support
    "LLVMAnalysis",
    "LLVMProfileData",
    "LLVMObject",
    "LLVMTextAPI",
    "LLVMBinaryFormat",
    "LLVMRemarks",
    "LLVMSupport",
    "LLVMDemangle",
    # Debug info
    "LLVMDebugInfoDWARF",
    "LLVMDebugInfoPDB",
    "LLVMDebugInfoMSF",
    "LLVMDebugInfoBTF",
    "LLVMDebugInfoCodeView",
    "LLVMSymbolize",
    # Frontend
    "LLVMFrontendOpenMP",
    "LLVMFrontendOffloading",
    # Misc
    "LLVMHipStdPar",
    "LLVMWindowsDriver",
    "LLVMOption",
]


def download_tarball(platform, dest_dir):
    """Download official LLVM tarball for a platform."""
    asset = PLATFORMS[platform]
    url = f"{RELEASE_URL}/{asset}"
    dest = os.path.join(dest_dir, asset)

    if os.path.exists(dest):
        print(f"  Already downloaded: {asset}")
        return dest

    print(f"  Downloading {asset}...")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"\r  {mb_down:.0f}/{mb_total:.0f} MB ({pct}%)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()  # newline after progress
    return dest


def repackage(platform, work_dir, output_dir):
    """Download, selectively extract needed files, and create minimal tarball."""
    is_windows = "windows" in platform
    lib_ext = ".lib" if is_windows else ".a"
    lib_prefix = "" if is_windows else "lib"

    extract_dir = EXTRACT_DIRS[platform]
    stage_name = f"llvm-{LLVM_VERSION}-{platform}"
    output_name = f"{stage_name}.tar.xz"
    output_path = os.path.join(output_dir, output_name)

    print(f"\n=== Repackaging {platform} ===")

    # Download
    tarball = download_tarball(platform, work_dir)

    # Build set of paths we want to extract
    wanted = set()
    for lib in REQUIRED_LIBS:
        wanted.add(f"{extract_dir}/lib/{lib_prefix}{lib}{lib_ext}")

    # C++ runtime static libs (Linux/macOS only)
    RT_SUBDIRS = {
        "x86_64-linux": "x86_64-unknown-linux-gnu",
        "aarch64-linux": "aarch64-unknown-linux-gnu",
        "aarch64-macos": "aarch64-apple-darwin",
    }
    rt_subdir = RT_SUBDIRS.get(platform)
    if rt_subdir:
        for lib in ["libc++.a", "libc++abi.a", "libunwind.a"]:
            wanted.add(f"{extract_dir}/lib/{rt_subdir}/{lib}")

    # llvm-c headers prefix (match anything under include/llvm-c/)
    headers_prefix = f"{extract_dir}/include/llvm-c/"

    # Stream through the tarball, extracting only what we need
    print(f"  Extracting {len(wanted)} libs + headers from tarball...")
    extracted_count = 0
    total_size = 0

    with tarfile.open(tarball, "r:xz") as src:
        with tarfile.open(output_path, "w:xz") as dst:
            for member in src:
                # Check if this member is one we want
                if member.name in wanted or member.name.startswith(headers_prefix):
                    # Rewrite the path: replace extract_dir prefix with stage_name
                    member.name = stage_name + member.name[len(extract_dir) :]
                    if member.isfile():
                        fileobj = src.extractfile(member)
                        dst.addfile(member, fileobj)
                        total_size += member.size
                        extracted_count += 1
                    elif member.isdir():
                        dst.addfile(member)

    # Verify by re-reading the output tarball
    missing = []
    with tarfile.open(output_path, "r:xz") as tf:
        members = set(m.name for m in tf.getmembers())
    for lib in REQUIRED_LIBS:
        if f"{stage_name}/lib/{lib_prefix}{lib}{lib_ext}" not in members:
            missing.append(lib)

    if missing:
        print(f"  WARNING: Missing libs: {missing}")

    output_size = os.path.getsize(output_path) / 1024 / 1024
    total_size_mb = total_size / 1024 / 1024
    lib_count = len(REQUIRED_LIBS) - len(missing)

    print(
        f"  Done: {lib_count} libs, {total_size_mb:.1f} MB uncompressed, {output_size:.1f} MB compressed"
    )
    print(f"  Output: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Repackage LLVM libs for zgram")
    parser.add_argument(
        "platforms", nargs="*", help="Platforms to repackage (default: all)"
    )
    parser.add_argument("--list", action="store_true", help="List available platforms")
    parser.add_argument(
        "--work-dir", default=None, help="Working directory for downloads/extraction"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory for tarballs"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep downloaded/extracted files (skip cleanup)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available platforms:")
        for name, asset in PLATFORMS.items():
            print(f"  {name:20s} <- {asset}")
        return

    platforms = args.platforms or list(PLATFORMS.keys())
    for p in platforms:
        if p not in PLATFORMS:
            print(f"Unknown platform: {p}")
            print(f"Available: {', '.join(PLATFORMS.keys())}")
            sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = args.work_dir or tempfile.mkdtemp(prefix="zgram-llvm-")
    output_dir = args.output_dir or os.path.join(script_dir, "dist")
    os.makedirs(output_dir, exist_ok=True)

    print(f"LLVM version: {LLVM_VERSION}")
    print(f"Work dir: {work_dir}")
    print(f"Output dir: {output_dir}")

    outputs = []
    for platform in platforms:
        path = repackage(platform, work_dir, output_dir)
        outputs.append((platform, path))

    print("\n=== Summary ===")
    for platform, path in outputs:
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  {os.path.basename(path):45s} {size:.1f} MB")

    # Cleanup work dir
    if not args.keep:
        print(f"\nCleaning up work dir: {work_dir}")
        shutil.rmtree(work_dir)
    else:
        print(f"\nKept work dir: {work_dir}")


if __name__ == "__main__":
    main()
