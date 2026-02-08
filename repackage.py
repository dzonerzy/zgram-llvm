#!/usr/bin/env python3
"""
Repackage LLVM static libraries into minimal archives for zgram's JIT pipeline.

Downloads full LLVM builds from dzonerzy/llvm-zig (built with zig cc),
extracts only the required libs and headers, and creates smaller tarballs.

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
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

LLVM_VERSION = "21.1.8"

# All platforms use llvm-zig builds (cross-compiled with zig cc)
LLVM_ZIG_URL = f"https://github.com/dzonerzy/llvm-zig/releases/download/v{LLVM_VERSION}"

PLATFORMS = {
    "x86_64-linux": {
        "url": f"{LLVM_ZIG_URL}/llvm-{LLVM_VERSION}-x86_64-linux.tar.xz",
        "asset": f"llvm-{LLVM_VERSION}-x86_64-linux.tar.xz",
        "extract_dir": f"llvm-{LLVM_VERSION}-x86_64-linux",
    },
    "aarch64-linux": {
        "url": f"{LLVM_ZIG_URL}/llvm-{LLVM_VERSION}-aarch64-linux.tar.xz",
        "asset": f"llvm-{LLVM_VERSION}-aarch64-linux.tar.xz",
        "extract_dir": f"llvm-{LLVM_VERSION}-aarch64-linux",
    },
    "x86_64-windows": {
        "url": f"{LLVM_ZIG_URL}/llvm-{LLVM_VERSION}-x86_64-windows.tar.xz",
        "asset": f"llvm-{LLVM_VERSION}-x86_64-windows.tar.xz",
        "extract_dir": f"llvm-{LLVM_VERSION}-x86_64-windows",
    },
    "aarch64-windows": {
        "url": f"{LLVM_ZIG_URL}/llvm-{LLVM_VERSION}-aarch64-windows.tar.xz",
        "asset": f"llvm-{LLVM_VERSION}-aarch64-windows.tar.xz",
        "extract_dir": f"llvm-{LLVM_VERSION}-aarch64-windows",
    },
    "aarch64-macos": {
        "url": f"{LLVM_ZIG_URL}/llvm-{LLVM_VERSION}-aarch64-macos.tar.xz",
        "asset": f"llvm-{LLVM_VERSION}-aarch64-macos.tar.xz",
        "extract_dir": f"llvm-{LLVM_VERSION}-aarch64-macos",
    },
}

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
    # New in LLVM 21 (split from LLVMCodeGen and LLVMDebugInfoDWARF)
    "LLVMCGData",
    "LLVMDebugInfoDWARFLowLevel",
]


def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        print(f"\r  {mb_down:.0f}/{mb_total:.0f} MB ({pct}%)", end="", flush=True)


def download(url, asset, dest_dir):
    """Download a file if not already present."""
    dest = os.path.join(dest_dir, asset)
    if os.path.exists(dest):
        print(f"  Already downloaded: {asset}")
        return dest
    print(f"  Downloading {asset}...")
    req = urllib.request.Request(url, headers={"User-Agent": "zgram-llvm/1.0"})
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 256 * 1024
        with open(dest, "wb") as out:
            while True:
                buf = resp.read(block_size)
                if not buf:
                    break
                out.write(buf)
                downloaded += len(buf)
                if total_size > 0:
                    pct = min(100, downloaded * 100 // total_size)
                    mb_down = downloaded / 1024 / 1024
                    mb_total = total_size / 1024 / 1024
                    print(
                        f"\r  {mb_down:.0f}/{mb_total:.0f} MB ({pct}%)",
                        end="",
                        flush=True,
                    )
    print()
    return dest


def repackage(platform, work_dir, output_dir):
    """Download, extract needed libs/headers, and create minimal tarball."""
    pinfo = PLATFORMS[platform]
    extract_dir = pinfo["extract_dir"]
    stage_name = f"llvm-{LLVM_VERSION}-{platform}"
    stage_dir = os.path.join(work_dir, stage_name)
    output_name = f"{stage_name}.tar.xz"
    output_path = os.path.join(output_dir, output_name)

    print(f"\n=== Repackaging {platform} ===")

    # Download LLVM tarball from llvm-zig
    tarball = download(pinfo["url"], pinfo["asset"], work_dir)

    # Build set of lib paths we want to extract
    wanted_libs = set()
    for lib in REQUIRED_LIBS:
        wanted_libs.add(f"{extract_dir}/lib/lib{lib}.a")

    # Header prefixes to extract
    header_prefixes = [
        f"{extract_dir}/include/llvm-c/",
        f"{extract_dir}/include/llvm/Config/",
    ]

    # Extract needed files to staging directory
    print(f"  Extracting {len(wanted_libs)} libs + headers from tarball...")
    os.makedirs(stage_dir, exist_ok=True)
    os.makedirs(os.path.join(stage_dir, "lib"), exist_ok=True)

    extracted_count = 0
    with tarfile.open(tarball, "r:xz") as src:
        for member in src:
            is_wanted_lib = member.name in wanted_libs
            is_wanted_header = any(member.name.startswith(p) for p in header_prefixes)
            if (is_wanted_lib or is_wanted_header) and member.isfile():
                rel_path = member.name[len(extract_dir) + 1 :]
                dest_path = os.path.join(stage_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as out:
                    out.write(src.extractfile(member).read())
                extracted_count += 1

    print(f"  Extracted {extracted_count} files")

    # Verify all required libs are present
    missing = []
    for lib in REQUIRED_LIBS:
        if not os.path.exists(os.path.join(stage_dir, "lib", f"lib{lib}.a")):
            missing.append(lib)
    if missing:
        print(f"  WARNING: Missing libs: {missing}")

    # Create output tarball
    print(f"  Creating {output_name}...")
    subprocess.run(
        ["tar", "cJf", output_path, "-C", work_dir, stage_name],
        check=True,
        env={**os.environ, "XZ_OPT": "-T0 -6"},
    )

    shutil.rmtree(stage_dir)

    output_size = os.path.getsize(output_path) / 1024 / 1024
    lib_count = len(REQUIRED_LIBS) - len(missing)
    print(f"  Done: {lib_count} libs, {output_size:.1f} MB compressed")
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
        for name, pinfo in PLATFORMS.items():
            print(f"  {name:20s} <- {pinfo['asset']}")
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

    if not args.keep:
        print(f"\nCleaning up work dir: {work_dir}")
        shutil.rmtree(work_dir)
    else:
        print(f"\nKept work dir: {work_dir}")


if __name__ == "__main__":
    main()
