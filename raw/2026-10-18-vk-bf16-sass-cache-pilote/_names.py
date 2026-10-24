"""_names.py — pipeline names for the extracted SASS: the VK_KHR_pipeline_executable_properties stats (name, Binary Size, Register Count) matched
by binary size; a size shared by several pipelines lists every candidate name."""
import os, re, collections
here = os.path.dirname(os.path.abspath(__file__))
for m in ["0.8B", "4B"]:
    by = collections.defaultdict(list)
    for l in open(os.path.join(here, "..", "2026-10-18-vk-bf16-statistiques-pipelines", f"stats-{m}.txt")):
        mt = re.match(r"(\S+) \| Register Count = (\d+) Binary Size = (\d+)", l)
        if mt: by[int(mt.group(3))].append(f"{mt.group(1)} ({mt.group(2)} reg)")
    print(f"== {m}")
    for l in open(os.path.join(here, f"glcache-{m}", "sass2.log")):
        mt = re.match(r"(\S+) size (\d+) instr (\d+) rc (\d+) ?(.*)", l)
        names = by.get(int(mt.group(2)), ["?"])
        print(f"{mt.group(1)} | {mt.group(2)} B | {mt.group(3)} instr | {mt.group(5)} | {' / '.join(sorted(set(names)))}")
