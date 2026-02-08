#!/usr/bin/env python3
"""
Repackage LLVM static libraries into minimal archives for zgram's JIT pipeline.

Linux/macOS: Downloads official LLVM release, extracts needed libs, converts
LTO bitcode to native ELF (Linux), and bundles portable libstdc++.a from
Bootlin musl toolchains.

Windows: Downloads pre-built MinGW LLVM libs from dzonerzy/llvm-windows-zig
(built with zig cc, uses libc++ ABI compatible with Zig's linkLibCpp).

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

# Map from our standardized name -> (release URL base, asset name, extract dir)
# Linux/macOS use official LLVM releases
# Windows uses our custom MinGW build from llvm-windows-zig (built with zig cc)
OFFICIAL_URL = (
    f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}"
)
MINGW_URL = f"https://github.com/dzonerzy/llvm-windows-zig/releases/download/llvm-{LLVM_VERSION}"

PLATFORMS = {
    "x86_64-linux": {
        "url": f"{OFFICIAL_URL}/LLVM-{LLVM_VERSION}-Linux-X64.tar.xz",
        "asset": f"LLVM-{LLVM_VERSION}-Linux-X64.tar.xz",
        "extract_dir": f"LLVM-{LLVM_VERSION}-Linux-X64",
    },
    "aarch64-linux": {
        "url": f"{OFFICIAL_URL}/LLVM-{LLVM_VERSION}-Linux-ARM64.tar.xz",
        "asset": f"LLVM-{LLVM_VERSION}-Linux-ARM64.tar.xz",
        "extract_dir": f"LLVM-{LLVM_VERSION}-Linux-ARM64",
    },
    "x86_64-windows": {
        "url": f"{MINGW_URL}/llvm-{LLVM_VERSION}-mingw-x86_64-windows.tar.xz",
        "asset": f"llvm-{LLVM_VERSION}-mingw-x86_64-windows.tar.xz",
        "extract_dir": f"llvm-{LLVM_VERSION}-mingw-x86_64-windows",
    },
    "aarch64-macos": {
        "url": f"{OFFICIAL_URL}/LLVM-{LLVM_VERSION}-macOS-ARM64.tar.xz",
        "asset": f"LLVM-{LLVM_VERSION}-macOS-ARM64.tar.xz",
        "extract_dir": f"LLVM-{LLVM_VERSION}-macOS-ARM64",
    },
}

# Bootlin musl toolchains for portable libstdc++.a (Linux only)
BOOTLIN_TOOLCHAINS = {
    "x86_64-linux": {
        "url": "https://toolchains.bootlin.com/downloads/releases/toolchains/x86-64/tarballs/x86-64--musl--stable-2025.08-1.tar.xz",
        "asset": "x86-64--musl--stable-2025.08-1.tar.xz",
        "libstdcxx": "x86_64-buildroot-linux-musl/sysroot/usr/lib/libstdc++.a",
        "libgcc_eh": "lib/gcc/x86_64-buildroot-linux-musl/14.3.0/libgcc_eh.a",
    },
    "aarch64-linux": {
        "url": "https://toolchains.bootlin.com/downloads/releases/toolchains/aarch64/tarballs/aarch64--musl--stable-2025.08-1.tar.xz",
        "asset": "aarch64--musl--stable-2025.08-1.tar.xz",
        "libstdcxx": "aarch64-buildroot-linux-musl/sysroot/usr/lib/libstdc++.a",
        "libgcc_eh": "lib/gcc/aarch64-buildroot-linux-musl/14.3.0/libgcc_eh.a",
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
    urllib.request.urlretrieve(url, dest, reporthook=report_progress)
    print()
    return dest


def _get_llvm_tools(work_dir):
    """Download and extract clang + llvm-ar from the x86_64-linux LLVM release.

    These are needed to convert LTO bitcode to native code. We always use the
    x86_64-linux tools since that's what CI runners (and most dev machines) run.
    Returns the path to the directory containing the tool binaries.
    """
    tools_dir = os.path.join(work_dir, "llvm-tools")
    if os.path.exists(os.path.join(tools_dir, "clang")):
        return tools_dir  # Already extracted

    x86_info = PLATFORMS["x86_64-linux"]
    tarball = download(x86_info["url"], x86_info["asset"], work_dir)
    x86_extract_dir = x86_info["extract_dir"]

    # clang is a symlink to clang-{major}, so extract the real binary
    llvm_major = LLVM_VERSION.split(".")[0]
    wanted_bins = {
        f"{x86_extract_dir}/bin/clang-{llvm_major}": "clang",
        f"{x86_extract_dir}/bin/llvm-ar": "llvm-ar",
    }

    os.makedirs(tools_dir, exist_ok=True)
    print(f"  Extracting LLVM {LLVM_VERSION} tools (clang-{llvm_major}, llvm-ar)...")
    with tarfile.open(tarball, "r:xz") as tf:
        for member in tf:
            if member.name in wanted_bins and member.isfile():
                dest = os.path.join(tools_dir, wanted_bins[member.name])
                with open(dest, "wb") as out:
                    out.write(tf.extractfile(member).read())
                os.chmod(dest, 0o755)

    return tools_dir


def convert_lto_lib(lib_path, clang="clang", llvm_ar="llvm-ar", target=None):
    """Convert a .a file containing LTO bitcode objects to native ELF objects.

    Official LLVM 21 Linux releases use -DLLVM_ENABLE_LTO=Thin, so the .a files
    contain LLVM IR bitcode instead of native machine code. This converts them.
    """

    work = lib_path + ".convert"
    os.makedirs(work, exist_ok=True)

    # Extract all .o files
    subprocess.run(
        [llvm_ar, "x", os.path.abspath(lib_path)],
        cwd=work,
        check=True,
        capture_output=True,
    )

    obj_files = []
    for root, _, files in os.walk(work):
        for f in files:
            if f.endswith(".o") or f.endswith(".obj"):
                obj_files.append(os.path.join(root, f))

    if not obj_files:
        shutil.rmtree(work)
        return

    # Convert each .o from bitcode to native, skip already-native objects.
    # LLVM bitcode files start with magic bytes 'BC' (0x42 0x43).
    native_objs = []
    bitcode_objs = []
    for obj in obj_files:
        with open(obj, "rb") as f:
            magic = f.read(2)
        if magic == b"BC":
            bitcode_objs.append(obj)
        else:
            native_objs.append(obj)  # Already native (e.g. hand-written asm)

    if not bitcode_objs:
        shutil.rmtree(work)
        return  # All objects were already native

    # Compile bitcode objects in parallel using threads (subprocess-bound work)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _convert_one(obj):
        native = obj + ".native"
        cmd = [clang, "-x", "ir", "-c", "-O2", "-o", native, obj]
        if target:
            cmd.extend(["--target=" + target])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return (False, os.path.basename(obj), result.stderr.strip())
        return (True, native, "")

    nproc = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        results = list(executor.map(_convert_one, bitcode_objs))

    for ok, info, err in results:
        if not ok:
            print(f"    ERROR converting {info}: {err}")
            raise RuntimeError(f"Failed to convert {info}")
        native_objs.append(info)

    # Repack into .a
    os.remove(lib_path)
    subprocess.run(
        [llvm_ar, "rcs", os.path.abspath(lib_path)] + native_objs,
        check=True,
        capture_output=True,
    )
    shutil.rmtree(work)


def repackage(platform, work_dir, output_dir):
    """Download, extract, convert, and create minimal tarball."""
    is_windows = "windows" in platform
    is_linux = "linux" in platform
    # All platforms now use lib*.a (MinGW for Windows, ELF for Linux/macOS)
    lib_prefix = "lib"
    lib_ext = ".a"

    pinfo = PLATFORMS[platform]
    extract_dir = pinfo["extract_dir"]
    stage_name = f"llvm-{LLVM_VERSION}-{platform}"
    stage_dir = os.path.join(work_dir, stage_name)
    output_name = f"{stage_name}.tar.xz"
    output_path = os.path.join(output_dir, output_name)

    print(f"\n=== Repackaging {platform} ===")

    # Download LLVM tarball
    tarball = download(pinfo["url"], pinfo["asset"], work_dir)

    # Build set of paths we want to extract
    wanted_libs = set()
    for lib in REQUIRED_LIBS:
        wanted_libs.add(f"{extract_dir}/lib/{lib_prefix}{lib}{lib_ext}")

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
                # Rewrite path: replace extract_dir with stage_dir
                rel_path = member.name[len(extract_dir) + 1 :]
                dest_path = os.path.join(stage_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as out:
                    out.write(src.extractfile(member).read())
                extracted_count += 1

    print(f"  Extracted {extracted_count} files")

    # Linux: convert LTO bitcode .a files to native ELF
    if is_linux:
        # Need clang 21 + llvm-ar 21 to read LLVM 21 LTO bitcode.
        # Always use x86_64-linux tools (CI runners are x86_64).
        tools_dir = _get_llvm_tools(work_dir)
        bundled_clang = os.path.join(tools_dir, "clang")
        bundled_ar = os.path.join(tools_dir, "llvm-ar")
        print(
            f"  Converting LTO bitcode libs to native ELF (using LLVM {LLVM_VERSION} clang)..."
        )
        clang_target = {
            "x86_64-linux": "x86_64-linux-gnu",
            "aarch64-linux": "aarch64-linux-gnu",
        }.get(platform)
        lib_dir = os.path.join(stage_dir, "lib")
        converted = 0
        for lib in REQUIRED_LIBS:
            lib_path = os.path.join(lib_dir, f"lib{lib}.a")
            if os.path.exists(lib_path):
                convert_lto_lib(
                    lib_path,
                    clang=bundled_clang,
                    llvm_ar=bundled_ar,
                    target=clang_target,
                )
                converted += 1
        print(f"  Converted {converted} libs")

    # Linux: add portable libstdc++.a + libgcc_eh.a from Bootlin musl toolchain
    if platform in BOOTLIN_TOOLCHAINS:
        bt = BOOTLIN_TOOLCHAINS[platform]
        print(f"  Downloading Bootlin musl toolchain...")
        bt_tarball = download(bt["url"], bt["asset"], work_dir)

        print(f"  Extracting libstdc++.a and libgcc_eh.a...")
        bt_extract_dir = bt["asset"].replace(".tar.xz", "")
        with tarfile.open(bt_tarball, "r:xz") as tf:
            for member in tf:
                rel = member.name
                # Strip the top-level dir from tarball
                if "/" in rel:
                    after_top = "/".join(rel.split("/")[1:])
                else:
                    continue
                if after_top == bt["libstdcxx"] or after_top == bt["libgcc_eh"]:
                    dest_name = os.path.basename(after_top)
                    dest_path = os.path.join(stage_dir, "lib", dest_name)
                    with open(dest_path, "wb") as out:
                        out.write(tf.extractfile(member).read())
                    print(f"    Extracted {dest_name}")

    # Verify all required libs are present
    missing = []
    for lib in REQUIRED_LIBS:
        if not os.path.exists(
            os.path.join(stage_dir, "lib", f"{lib_prefix}{lib}{lib_ext}")
        ):
            missing.append(lib)
    if missing:
        print(f"  WARNING: Missing libs: {missing}")

    # Create output tarball (use system tar with parallel xz for speed)
    print(f"  Creating {output_name}...")
    subprocess.run(
        ["tar", "cJf", output_path, "-C", work_dir, stage_name],
        check=True,
        env={**os.environ, "XZ_OPT": "-T0 -6"},
    )

    # Cleanup staging dir
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

    # Cleanup work dir
    if not args.keep:
        print(f"\nCleaning up work dir: {work_dir}")
        shutil.rmtree(work_dir)
    else:
        print(f"\nKept work dir: {work_dir}")


if __name__ == "__main__":
    main()
