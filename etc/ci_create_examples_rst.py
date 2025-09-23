import os
import re
import shutil
from subprocess import PIPE, Popen

# check that pandoc is available and the version is sufficient
args = ("pandoc", "--version")
proc = Popen(args, stdout=PIPE, stderr=PIPE, cwd=".")
stdout, stderr = proc.communicate()
msg = ""
if stdout:
    for line in stdout.decode("utf-8").splitlines():
        if "pandoc" in line:
            t = line.split()[1]
            version = []
            for v in t.split("."):
                version.append(int(v))
            version = tuple(version)
            if version[0:2] < (2, 11):
                msg = "pandoc 2.11 or higher must be available"
            break
if stderr:
    msg = stderr.decode("utf-8")
if len(msg) > 0:
    raise RuntimeError(msg)
else:
    print(f"\npandoc version: {t}\n")

# parse examples from body.tex (in order)
pth = os.path.join("..", "doc", "body.tex")
with open(pth) as f:
    lines = f.read()
examples = []
ex_regex = re.compile(r"\\input{sections/(.*?)\}")
for v in ex_regex.findall(lines):
    if "ex-" in v:
        examples.append(v.replace(".tex", ""))

# get list of example problems
ex_dirs = []
ex_pth = os.path.join("..", "examples")
for name in sorted(os.listdir(ex_pth)):
    pth = os.path.join(ex_pth, name)
    if os.path.isdir(pth):
        ex_dirs.append(name)

# create examples rst
print(f"creating...'{pth}'")
with open(os.path.join("..", ".doc", "examples.rst"), "w") as f:
    lines = "Example descriptions\n"
    lines += (len(lines) - 1) * "-" + "\n\n"
    lines += "An overview of each MODFLOW 6 example problem.\n\n"
    f.write(lines)

    lines = ".. toctree::\n"
    lines += "   :numbered:\n"
    lines += "   :maxdepth: 1\n\n"
    for base_name in examples:
        lines += f"   _examples/{base_name}.rst\n"
    lines += "\n\n"
    f.write(lines)

# create rtd examples directory
dst = os.path.join("..", ".doc", "_examples")
print(f"cleaning and creating...'{dst}'")
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)

# read base latex file
pth = os.path.join("..", "doc", "mf6examples.tex")
print(f"reading...'{pth}'")
with open(pth) as f:
    orig_latex = f.readlines()

latex_tag = "\\input{./body.tex}"
frontmatter_tag = "\\input{./frontmatter.tex}"
doc_pth = os.path.join("..", "doc")
for ex in examples:
    print(f"creating restructured text file for {ex} example")
    src = os.path.join("..", "doc", "ex.tex")
    with open(src, "w") as f:
        for line in orig_latex:
            if latex_tag in line:
                new_tag = f"\\input{{sections/{ex}.tex}}"
                line = line.replace(latex_tag, new_tag)
            elif frontmatter_tag in line:
                line = line.replace(frontmatter_tag, "")
            f.write(line)

    # create restructured text file for example using using pandoc
    dst = os.path.join("..", ".doc", "_examples", f"{ex}.rst")
    print(f"running pandoc to create {dst}")
    args = (
        "pandoc",
        "-s",
        "--citeproc",
        "-f",
        "latex",
        "-t",
        "rst",
        "--bibliography=mf6examples.bib",
        "--csl=wrr.csl",
        "ex.tex",
        "-o",
        dst,
    )

    print(" ".join(args))
    proc = Popen(args, stdout=PIPE, stderr=PIPE, cwd=doc_pth)
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout.decode("utf-8"))
    if stderr:
        print("Errors:")
        print(stderr.decode("utf-8"))

    # read restructured text file as a string
    print(f"reading...'{dst}'")
    with open(dst, "r", encoding="utf-8") as file:
        lines = file.read()

    # find equation labels in lines
    ex_regex = re.compile(r"\\label{(.*?)\}")
    replace_eq_labels = {}
    eq_labels = []
    for v in ex_regex.findall(lines):
        tag = f"\\label{{{v}}}"
        label = f"   :label: {v.replace(':', '-')}"
        eq_labels.append(label)
        tag = f"`[{v}] <#{v}>`__"
        replace_eq_labels[tag] = f":eq:`{v.replace(':', '-')}`"

    # find figure references in lines
    ex_regex = re.compile(r"\`(.*?)\`__")
    ex_tag = re.compile(r"\#(.*?)\>")
    fig_tab_refs = {}
    for v in ex_regex.findall(lines):
        tag0 = f"`{v}`__"
        for tag in ex_tag.findall(tag0):
            if tag0 not in list(fig_tab_refs.keys()):
                fig_tab_refs[tag0] = f":numref:`{tag}`"

    # read restructured text file for example
    print(f"reading...'{dst}'")
    with open(dst, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # editing restructured text file for example
    print(f"editing...'{dst}'")
    with open(dst, "w", encoding="utf-8") as f:
        write_line = True
        in_reference = False
        eq_no = 0
        in_table = False
        table_name = None
        for idx, line in enumerate(lines):
            # skip the title
            if idx < 6:
                continue

            # replace latex equation labels
            for key, value in replace_eq_labels.items():
                if key in line:
                    line = line.replace(key, value)

            # replace figure and table references
            for key, value in fig_tab_refs.items():
                if key in line:
                    line = line.replace(key, value)

            tag = "\xa0"  # non-breaking space
            if tag in line:
                line = line.replace(tag, " ")

            tag = "tab:ex-"
            if tag in line:
                line = line.replace(tag, "tab-ex-")

            tag = "fig:ex-"
            if tag in line:
                line = line.replace(tag, "fig-ex-")

            tag = "\\label{"
            if tag in line:
                continue

            tag = ".. math::"
            if tag in line:
                line = f"{tag}\n{eq_labels[eq_no]}\n"
                eq_no += 1

            tag = ".. figure:: ../figures/"
            if tag in line:
                line = line.replace(tag, ".. figure:: ../_images/")

            tag = ".. figure:: ../images/"
            if tag in line:
                line = line.replace(tag, ".. figure:: ../_images/")

            tag = ":alt:"
            if tag in line:
                write_line = False

            tag = ":name:"
            if not write_line and tag in line:
                write_line = True

            # table tabs
            tags = (
                "Table :numref:",
                "table :numref:",
                "Tables :numref:",
            )
            for tag in tags:
                if tag in line:
                    line = line.replace(tag, ":numref:")

            # figure tags
            tags = (
                "fig. :numref:",
                "Figure :numref:",
                "figure :numref:",
                "figures :numref:",
            )
            for tag in tags:
                if tag in line:
                    line = line.replace(tag, ":numref:")

            tag = ".. container::"
            if tag == line.strip():
                in_table = True
                continue

            if in_table:
                if len(line.lstrip()) == len(line):
                    line = "\n" + line
                    in_table = False

            tag = ":name: tab-ex-"
            if tag in line:
                table_name = line.split()[1]
                continue

            tag = ".. table::"
            if tag in line:
                line = f".. _{table_name}:\n\n" + line.lstrip() + "\n"

            tag = ".. container:: references"
            if tag in line:
                in_reference = True
                line = "References Cited\n----------------\n\n"

            if in_reference:
                tag = ".. container::"
                if tag in line:
                    continue

                tag = ":name:"
                if tag in line:
                    continue

                if line.startswith("      "):
                    line = line.lstrip()

            if write_line:
                f.write(line)

        # Jupyter Notebook section
        ex_root = ex.replace(".tex", "")

        # write Jupyter Notebook section
        line = "\n\nJupyter Notebook\n"
        line += "----------------\n\n"
        line += (
            "The Jupyter notebook used to create the MODFLOW 6 input files\n"
            "for this example and post-process the results is:\n\n"
        )
        line += f"* `{ex_root} <../_notebooks/{ex_root}.html>`_\n"
        line += "\n"

        # Check to see if there is a gif with the same name as the example name
        # (e.g. ex-gwt-saltlake.gif) and if so, then assume this is an animated
        # gif and add an Animation section to the restructured text file
        fname = f"{ex}.gif"
        if fname in os.listdir("../figures"):
            line = "\n\nAnimation\n"
            line += "----------------\n\n"
            line += "Animation of model results:\n\n"
            line += f".. image:: ../_images/{fname}"
        line += "\n"

        # Write the restructured text lines and close the file
        f.write(line)

    # clean up temporary latex file
    if os.path.isfile(src):
        os.remove(src)

# create rtd figure directory
dst = os.path.join("..", ".doc", "_images")
print(f"cleaning and creating...'{dst}'")
if os.path.isdir(dst):
    shutil.rmtree(dst)
os.makedirs(dst, exist_ok=True)

# copy figures to rtd directory
src_dirs = (os.path.join("..", "figures"), os.path.join("..", "images"))
for src_dir in src_dirs:
    file_names = [
        file_name
        for file_name in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, file_name))
        and (file_name.endswith(".png") or file_name.endswith(".gif"))
    ]
    for file_name in file_names:
        src = os.path.join(src_dir, file_name)
        print(f"copy '{src}' -> '{dst}' directory")
        shutil.copy2(src, dst)
