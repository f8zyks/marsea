"""Patch RULER's variable_tracking.py (2026-09-22) by exact string replacement, like patch_ruler.py for niah.py.

Stock generate_chains draws C (H + 1) random variable names and, on a duplicate, APPENDS more names -- but then slices
the longer list in windows of H + 1, so the last chain comes out short and `this_vars[j+1]` raises IndexError.  The
one-shot example uses 3-letter names (17,576 of them): at 36 names a duplicate happens in ~4 % of draws, and the
generator dies on it (VT_L4096_C4_H8 for two of three seeds).  The names are drawn unique here.  Recorded as
third_party/RULER/marsea_vt.patch; ruler.ensure_vt_patched() re-applies on a miss."""
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party/RULER/scripts/data/synthetic/variable_tracking.py"
ORIG = SRC.with_suffix(".py.orig")
PATCH = ROOT / "third_party/RULER/marsea_vt.patch"
MARKER = "## MarSea-vt-patch-v1"


def apply(text: str) -> str:
    if MARKER in text:
        return text
    a = ("    vars_all = [''.join(random.choices(string.ascii_uppercase, k=k)).upper() for _ in range((num_hops+1) * num_chains)]\n"
         "    while len(set(vars_all)) < num_chains * (num_hops+1):\n"
         "        vars_all.append(''.join(random.choices(string.ascii_uppercase, k=k)).upper())\n")
    assert a in text, "generate_chains name draw not found"
    b = ("    " + MARKER + ": stock RULER appends on a duplicate and then slices the longer list in windows of\n"
         "    # num_hops + 1, so the last chain is short and `this_vars[j+1]` raises IndexError (~4 % of draws at 36 three-letter names)\n"
         "    vars_all = []\n"
         "    while len(vars_all) < num_chains * (num_hops+1):\n"
         "        v = ''.join(random.choices(string.ascii_uppercase, k=k)).upper()\n"
         "        if v not in vars_all:\n"
         "            vars_all.append(v)\n")
    return text.replace(a, b)


if __name__ == "__main__":
    src = SRC.read_text()
    if not ORIG.exists():
        ORIG.write_text(src)
    new = apply(ORIG.read_text())
    SRC.write_text(new)
    compile(new, str(SRC), "exec")
    res = subprocess.run(["diff", "-u", str(ORIG), str(SRC)], capture_output=True, text=True)
    PATCH.write_text(res.stdout.replace(str(ORIG), "a/scripts/data/synthetic/variable_tracking.py").replace(str(SRC), "b/scripts/data/synthetic/variable_tracking.py"))
    print(f"patched {SRC}; diff -> {PATCH} ({len(res.stdout.splitlines())} lines)")
