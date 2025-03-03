"""
Build LaTeX and Markdown tables for the PDF documentation and ReadtheDocs.

The files that are processed are the example python scripts (expy) and
the example directories (exdir) that are created from running those scripts.

There are a set of optional flags that take up to two additional arguments.

Flag1 and Flag2 are equivalent representations of the same flag.

For options that search for a subset of expy and exdir,
Arg1 and Arg2 are unix search pattern strings to filter the expy and exdir.
If either Arg1 or Arg2 is set to "None", then the search that it is meant
to perform is disabled and no result is returned.

Flag1 Flag2    Arg1 Arg2  Description

-v  --verbose    -   -    Indicates that while processing the expy and exdir,
                          print detailed file information.

-p  --print      -   -    If present, prints all found expy and exdir
                          and exits the program.

-f  --find     yes  opt  Prints expy and exdir that match a Unix
                         filename search pattern and exits the program.
                         Arg1 is the Unix filename matching pattern
                         to filter and print the expy and exdir names.
                         Arg2 is optional, and if present replaces Arg1 for
                         pattern matching exdir.

-l  --list     yes  opt  Prints in the shape of a single python list the
                         expy and exdir that match a Unix filename
                         search pattern and exits the program.
                         Arg1 is the Unix filename matching pattern
                         to filter and print the expy and exdir names.
                         Arg2 is optional, and if present replaces Arg1 for
                         pattern matching exdir.

-k  --keyword  yes  opt  Process the expy and exdir that match a Unix
                         filename search pattern.
                         Arg1 is the Unix filename matching pattern
                         to filter and print the expy and exdir names.
                         Arg2 is optional, and if present replaces Arg1 for
                         pattern matching exdir.

    Examples
    --------
    >>> process-scripts.py --print     # print to prompt all example problems.
    >>> process-scripts.py -f *-maw-*  # print to prompt MAW example problems.
    >>> process-scripts.py -k *-maw-*  # process all MAW example problems.
"""

import ast
import fnmatch
import os
import re
import sys
from pathlib import Path

import flopy
from modflow_devtools.latex import get_footer, get_header

# path to the example files
proj_root = Path(__file__).parents[1]
ex_pth = proj_root / "examples"

# only process python files starting with ex_
files = sorted(
    [file for file in os.listdir() if file.endswith(".py") and file.startswith("ex-")]
)

only_process_ex = []


def _replace_quotes(proc_str):
    for r in (("'", ""), ('"', "")):
        proc_str = proc_str.replace(*r)
    return proc_str


def table_standard_header(caption, label):
    col_widths = (0.5, 0.3)
    headings = ("Parameter", "Value")
    return get_header(caption, label, headings, col_widths=col_widths, center=False)


def table_scenario_header(caption, label):
    col_widths = (0.1, 0.25, 0.3, 0.15)
    headings = ("Scenario", "Scenario Name", "Parameter", "Value")
    return get_header(caption, label, headings, col_widths=col_widths, center=False)


def table_footer():
    return get_footer()


def make_tables():
    tab_pth = proj_root / "tables"
    if not os.path.isdir(tab_pth):
        os.makedirs(tab_pth)

    for file in files:
        print(f"processing...'{file}'")
        basename = os.path.splitext(os.path.basename(file))[0].replace("_", "-")
        # do a little processing
        with open(file) as f:
            txt = f.read()
        doc = ast.parse(txt, file)
        lines = txt.splitlines()

        # gather all assignments, organize by (starting) lineno and name
        assign_bylineno = {}
        assign_byname = {}
        for node in doc.body:
            if isinstance(node, ast.Assign):
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    assign_bylineno[node.lineno] = node
                    assign_byname[target.id] = node.value

        # parse optional parameters dictionary
        parameters = None
        if "parameters" in assign_byname:
            obj = assign_byname["parameters"]
            parameters = eval(ast.unparse(obj))

        # parse optional parameter units dictionary, if parameters are
        # specified in the script
        if parameters is not None:
            parameter_units = None
            if "parameter_units" in assign_byname:
                obj = assign_byname["parameter_units"]
                parameter_units = eval(ast.unparse(obj))

        # create scenario table if parameters are specified in the script
        if parameters is not None:
            tab_name = f"{basename}-scenario"
            caption = f"Scenario parameters for example {basename}."
            label = f"tab:{tab_name}"
            fpth = os.path.join(tab_pth, tab_name + ".tex")
            f = open(fpth, "w")

            f.write(table_scenario_header(caption, label))

            scenario_count = 0
            for scenario_name, value_dict in parameters.items():
                if scenario_count % 2 != 0:
                    row_color = "\t\t\\rowcolor{Gray}\n"
                else:
                    row_color = ""
                scenario_count += 1
                table_line = f"{scenario_count} & {scenario_name} & "
                for text, value in value_dict.items():
                    text_to_write = text.replace("_", "~")
                    units = ""
                    if parameter_units is not None:
                        try:
                            units = f" ({parameter_units[text]})"
                        except:
                            units = " (unknown)"
                    if len(table_line) > 0:
                        table_line += f"\t{text_to_write}{units} & {value}"
                    else:
                        table_line = f"\t& & {text_to_write}{units} & {value}"
                    table_line += " \\\\\n"
                    if len(row_color) > 0:
                        f.write(row_color)
                    f.write(table_line)
                    table_line = ""

            # write footer
            f.write(table_footer())

            # finalize table
            f.close()

        # tables
        table_number = 0
        scanning_table = False
        table_text = []
        table_value = []
        for lineno, line in enumerate(lines, 1):
            if line.lower().startswith("# model parameters"):
                scanning_table = True
                continue
            if scanning_table:
                if line.startswith("# "):
                    scanning_table = False
                if lineno in assign_bylineno:
                    obj = assign_bylineno[lineno]
                    # get comment from at least one line
                    for idx in range(obj.lineno - 1, obj.end_lineno):
                        table_line = lines[idx]
                        if (ipos := table_line.find("# ")) > 0:
                            table_text.append(table_line[ipos + 1 :].strip())
                            break
                    else:
                        scanning_table = False
                        continue
                    # get content on right side of "="
                    val = obj.value
                    if obj.lineno == obj.end_lineno:
                        value = line[val.col_offset : val.end_col_offset]
                    else:
                        # string may span more than one line
                        value = obj.value.value
                    if value.startswith("'") or value.startswith('"'):
                        value = _replace_quotes(value)
                    table_value.append(value)
            if not scanning_table and len(table_text) > 0:
                table_number += 1
                tab_name = f"{basename}-{table_number:02d}"
                caption = f"Model parameters for example {basename}."
                label = f"tab:{tab_name}"
                fpth = os.path.join(tab_pth, tab_name + ".tex")
                f = open(fpth, "w")

                f.write(table_standard_header(caption, label))

                # write table
                line_count = 0
                for text, value in zip(table_text, table_value):
                    if line_count % 2 != 0:
                        f.write("\t\\rowcolor{Gray}\n")
                    # replace minus signs with double minus signs for
                    # latex render (so minus sign "sticks" to number)
                    # but don't replace "e-" in scientific notation.
                    value = re.sub(
                        r"(?<!e)(-)\d*", lambda m: m.group(0).replace("-", "--"), value
                    )
                    f.write(f"\t{text} & {value} \\\\\n")
                    line_count += 1

                # write footer
                f.write(table_footer())

                # finalize table
                f.close()

                # reset these if there is more than one table
                table_text = []
                table_value = []


def get_ordered_examples(verbose=True):
    if verbose:
        print("creating a ordered list of examples from the LaTeX document")
    # get order of examples from body.text
    ex_regex = re.compile("\\\\input{sections/(.*?)\\}")
    ex_order = []
    pth = proj_root / "doc" / "body.tex"
    with open(pth) as f:
        lines = f.read()
    for v in ex_regex.findall(lines):
        ex_order.append(v.replace(".tex", ""))
    return ex_order


def get_examples_list(verbose=True):
    if verbose:
        print("creating a list of available examples")

    # examples to exclude
    exclude = ("ex-gwf-csub-p02c",)

    # get order of examples from body.text
    ex_order = get_ordered_examples(verbose)

    # get list of all example names
    ex_list = sorted(
        [p.stem for p in Path(ex_pth).glob("*") if p.is_dir() and p.name not in exclude]
    )

    # create final list with all of the examples in body.tex order
    ex_final = []
    for o in ex_order:
        for n in ex_list:
            if n.startswith(o) and n not in ex_final:
                ex_final.append(n)

    # add any orphans
    if len(ex_final) != len(ex_list):
        for n in ex_list:
            if n not in ex_final:
                ex_final.append(n)

    return ex_final


def get_examples_dict(verbose=False):
    print("creating dictionary with examples information")
    ex_list = get_examples_list()
    if len(only_process_ex) > 0:
        ex_list = [item for item in ex_list if item in only_process_ex]
    ex_dict = {}
    for ex_name in ex_list:
        namefiles = []
        dimensions = []
        paks = []
        simulation_paks = []
        rootDir = os.path.join(ex_pth, ex_name)
        for dirName, subdirList, fileList in os.walk(rootDir):
            if verbose:
                print(f"Found directory: {dirName}")
            for file_name in fileList:
                if file_name.lower() == "mfsim.nam":
                    if verbose:
                        msg = (
                            "  Found MODFLOW 6 simulation " + f"name file: {file_name}"
                        )
                        print(msg)
                    print(f"Using flopy to load {dirName}")
                    sim = flopy.mf6.MFSimulation.load(sim_ws=dirName, verbosity_level=0)
                    sim_paks = []
                    for pak in sim.sim_package_list:
                        pak_type = pak.package_abbr
                        if pak_type not in simulation_paks:
                            sim_paks.append(pak_type)
                    for model_name in sim.model_names:
                        model = sim.get_model(model_name)
                        namefiles.append(model.namefile)
                        dimensions.append(model.modelgrid.shape)
                        model_paks = []
                        for pak in model.packagelist:
                            pak_type = pak.package_type
                            if pak_type not in model_paks:
                                model_paks.append(pak_type)
                                if pak_type == "npf":
                                    if model.npf.xt3doptions.array:
                                        model_paks.append("xt3d")
                        simulation_paks.append(sim_paks)
                        paks.append(model_paks)
        # add packages for simulation to ex_paks dictionary
        ex_dict[ex_name] = {
            "namefiles": namefiles,
            "dimensions": dimensions,
            "paks": paks,
            "simulation_paks": simulation_paks,
        }
    return ex_dict


def build_md_tables(ex_dict):
    print("building markdown tables for ReadtheDocs")
    # determine the order of the examples from the LaTeX document
    ex_order = get_ordered_examples()

    # create a dictionary in the correct order
    ex_md = {}
    for ex in ex_order:
        for key, d in ex_dict.items():
            if ex in key:
                ex_md[key] = d

    # build a dictionary with all of the unique packages for a example
    ex_paks = {}
    for key, d in ex_md.items():
        paks = d["simulation_paks"][0]
        # add model packages that have not ben added yet
        for model_paks in d["paks"]:
            # this is the union of the two lists
            paks = list(set(paks) | set(model_paks))
        ex_paks[key] = paks

    # build dictionary with hyperlinks
    pak_link = {}
    for ex_name in ex_paks.keys():
        for ex_root in ex_order:
            if ex_root in ex_name:
                pak_link[ex_name] = f"[{ex_name}](_examples/{ex_root}.html)"
                break
        if ex_name not in list(pak_link.keys()):
            pak_link[ex_name] = ex_name

    # build list of unique packages
    pak_unique = []
    for ex_name, paks in ex_paks.items():
        for pak in paks:
            if pak not in pak_unique:
                pak_unique.append(pak)

    pak_dict = {}
    for pak in pak_unique:
        ex_list = []
        for ex_name, paks in ex_paks.items():
            if pak in paks:
                ex_list.append(pak_link[ex_name])
        pak_dict[pak] = ex_list

    rtd_link = "https://modflow6.readthedocs.io/en/latest/_mf6io/"
    pth = proj_root / ".doc" / "introduction.md"

    with open(pth, "w") as f:
        line = "### Introduction\n\n"
        line += (
            "This document describes example problems for MODFLOW 6. The examples "
            "demonstrate use of the capabilities for select components of "
            "MODFLOW 6. A pdf of the examples can be downloaded from "
            "[ReadtheDocs](https://modflow6-examples.readthedocs.io/en/latest/) "
            "or from the [current release](https://github.com/MODFLOW-ORG/"
            "modflow6-examples/releases/download/current/mf6examples.pdf) on "
            "[GitHub](https://github.com/MODFLOW-ORG/modflow6-examples/).\n\n"
            "Examples have been included for the MODFLOW 6 components "
            "summarized in the tables below.\n\n"
        )
        f.write(line)

        join_fmt = ", "
        header = (
            "| Package | Examples                                                           |\n"
            "|---------|--------------------------------------------------------------------|\n"
        )
        footer = "\n\n"

        # Discretization
        pak_link = {
            "dis": "gwf-dis.html",
            "disv": "gwf-disv.html",
            "disu": "gwf-disu.html",
        }
        line = "#### Discretization types\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        # Exchanges
        pak_link = {
            "gwfgwf": "exg-gwfgwf.html",
            "gwfgwt": "exg-gwfgwt.html",
            "gwfgwe": "exg-gwfgwe.html",
            "gwfprt": "exg-gwfprt.html",
        }
        line = "#### Exchanges\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        # Groundwater Flow
        pak_link = {
            "npf": "gwf-npf.html",
            "hfb": "gwf-hfb.html",
            "sto": "gwf-sto.html",
            "csub": "gwf-csub.html",
        }
        line = "#### Groundwater Flow Model Internal Flow Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        pak_link = {
            "rch": "gwf-rch.html",
            "rcha": "gwf-rcha.html",
            "evt": "gwf-evt.html",
            "evta": "gwf-evta.html",
            "chd": "gwf-chd.html",
            "drn": "gwf-drn.html",
            "ghb": "gwf-ghb.html",
            "riv": "gwf-riv.html",
            "wel": "gwf-wel.html",
            "buy": "gwf-buy.html",
            "vsc": "gwf-vsc.html",
        }
        line = "#### Groundwater Flow Model Standard Boundary Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        pak_link = {
            "lak": "gwf-lak.html",
            "sfr": "gwf-sfr.html",
            "maw": "gwf-maw.html",
            "uzf": "gwf-uzf.html",
            "mvr": "gwf-mvr.html",
        }
        line = "#### Groundwater Flow Model Advanced Boundary Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        # Groundwater Transport
        pak_link = {
            "adv": "gwt-adv.html",
            "dsp": "gwt-dsp.html",
            "mst": "gwt-mst.html",
            "ist": "gwt-ist.html",
            "fmi": "gwt-fmi.html",
        }
        line = "#### Groundwater Transport Model Internal Flow Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer + "\n\n"
        f.write(line)

        pak_link = {
            "ssm": "gwt-ssm.html",
            "src": "gwt-src.html",
            "cnc": "gwt-cnc.html",
        }
        line = "#### Groundwater Transport Model Standard Boundary Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        pak_link = {
            "lkt": "gwt-lkt.html",
            "sft": "gwt-sft.html",
            "mwt": "gwt-mwt.html",
            "uzt": "gwt-uzt.html",
            "mvt": "gwt-mvt.html",
        }
        line = "#### Groundwater Transport Model Advanced Boundary Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        # Energy Transport
        pak_link = {
            "cnd": "gwt-cnd.html",
            "est": "gwe-est.html",
        }
        line = "#### Groundwater Transport Model Internal Flow Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer + "\n\n"
        f.write(line)

        pak_link = {
            "esl": "gwe-esl.html",
            "ctp": "gwt-ctp.html",
        }
        line = "#### Groundwater Energy Transport Model Standard Boundary Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        pak_link = {
            "lke": "gwt-lke.html",
            "sfe": "gwt-sfe.html",
            "mwe": "gwt-mwe.html",
            "uze": "gwt-uze.html",
            "mve": "gwt-mve.html",
        }
        line = "#### Groundwater Energy Transport Model Advanced Boundary Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)

        # Particle Tracking
        pak_link = {
            "mip": "prt-mip.html",
            "prp": "prt-prp.html",
        }
        line = "#### Particle Tracking Model Packages\n\n"
        line += header
        for pak, value in pak_link.items():
            if pak in pak_dict.keys():
                line += f"| [{pak.upper()}]({rtd_link}{value}) |"
                line += f" {join_fmt.join(pak_dict[pak])} |\n"
        line += footer
        f.write(line)


def build_tex_tables(ex_dict):
    print("building LaTeX table for mf6examples.tex introduction")
    ex_order = get_ordered_examples()
    ex_tex = {}
    for idx, ex in enumerate(ex_order):
        for key, d in ex_dict.items():
            if ex in key:
                ex_number = [idx + 1] + [" " for i in range(len(d["paks"]) - 1)]
                d["ex_number"] = ex_number
                ex_tex[key] = d

    # build latex table for pdf document
    headings = (
        "Example",
        "Simulation",
        "Namefile(s)",
        "Grid \\newline Dimensions",
        "Packages",
    )
    col_widths = (0.10, 0.22, 0.25, 0.15, 0.28)
    caption = "List of example problems and simulation characteristics."
    label = "tab:ex-table"

    lines = get_header(caption, label, headings, col_widths=col_widths, firsthead=True)

    on_ex = 0
    for idx, (key, sim_dict) in enumerate(ex_tex.items()):
        for jdx, (ex_number, namefile, dimensions, model_paks, sim_paks) in enumerate(
            zip(
                sim_dict["ex_number"],
                sim_dict["namefiles"],
                sim_dict["dimensions"],
                sim_dict["paks"],
                sim_dict["simulation_paks"],
            )
        ):
            # union of sim_paks and model_paks to create unique list of packages
            paks = sim_paks
            for pak in model_paks:
                if pak not in paks:
                    paks.append(pak)

            # eliminate any duplicate packages
            paks = sorted(list(set(paks)))

            # separate examples by a line
            if isinstance(ex_number, int):
                if ex_number > on_ex:
                    on_ex = ex_number
                    include_idx = True
                    lines += "\t\t\\hline\n"
            if on_ex % 2 == 0:
                lines += "\t\t\\rowcolor{Gray}\n"
            lines += "\t\t"
            if include_idx:
                include_idx = False
                lines += f"{ex_number} & "
            else:
                lines += " & "
            if jdx == 0:
                lines += f"{key} & "
            else:
                lines += " & "
            lines += " {} & ".format(namefile.replace("_", "\\_"))
            lines += f" {dimensions} & "

            # packages
            pak_line = []
            for pak in paks:
                pak_line.append(pak.upper())
            lines += " ".join(pak_line) + " \\\\\n"

    lines += get_footer()

    # create table
    pth = proj_root / "tables" / "ex-table.tex"
    f = open(pth, "w")
    f.write(lines)
    f.close()


if __name__ == "__main__":
    verbose = False
    # parse command line arguments
    arg1 = None
    for it, arg in enumerate(sys.argv):
        if arg in ("-v", "--verbose"):
            verbose = True
        elif arg in ("-p", "--print"):
            print("\nList of all example scripts\n")
            for file in files:
                print(file)
            print("\n\nList of all available examples\n")
            for file in sorted(get_examples_list(False)):
                print(file)
            print()
            sys.exit(0)
        elif arg in ("-f", "--find", "-l", "--list", "-k", "-keyword"):
            # Setup Unix Style search pattern
            if "=" in arg:
                arg1 = arg[arg.find("=") + 1 :]
            else:
                it += 1
                try:
                    arg1 = sys.argv[it]
                except IndexError:
                    raise RuntimeError(
                        f"flag {arg} found, but it must be "
                        "followed by the unix-style search "
                        "expression."
                    )
            arg1 = arg1.replace("'", " ").replace('"', " ").strip()
            if arg1.lower() == "none":
                arg1 = "\x0fnull\x0e\x15\x17"  # always fail search
            it += 1
            try:
                arg2 = sys.argv[it]
                arg2 = arg2.replace("'", " ").replace('"', " ").strip()
                if arg2.lower() == "none":
                    arg2 = "\x0fnull\x0e\x15\x17"  # always fail search
            except IndexError:
                arg2 = arg1
            break

    if arg1 is not None:  # Found unix search flag
        if arg in ("-f", "--find"):
            print(f"\nList of example scripts that match {arg1}\n")
            for file in fnmatch.filter(files, arg1):
                print(file)
            files.sort()
            print(f"\nList of available examples that match {arg2}\n")
            for file in sorted(fnmatch.filter(get_examples_list(False), arg1)):
                print(file)
            print()
            sys.exit(0)
        else:  # if arg in ("-l", "--list", "-k", "-keyword"):
            lst = []
            for file in fnmatch.filter(files, arg1):
                lst.append(file)
            lst.sort()
            for file in sorted(fnmatch.filter(get_examples_list(False), arg1)):
                lst.append(file)
            lst = "[" + ", ".join(lst) + "]"
            if arg in ("-l", "--list"):
                print(
                    "\nPython list of example scripts that match "
                    f"{arg1} and example directories that match {arg2}\n\n"
                    f"{lst}\n"
                )
                sys.exit(0)
            sys.argv = [sys.argv[0], lst]

    # check if only doing a specified files
    if len(sys.argv) > 1:
        cmd_line = " ".join(sys.argv[1:])
        st = cmd_line.find("[") + 1
        sp = cmd_line.find("]")
        if st > 0:
            cmd_line = cmd_line[st:sp]  # becomes "" if [ not found
            only_process_ex = [
                item.replace("'", " ").replace('"', " ").strip()
                for item in cmd_line.split(",")
            ]

        if len(only_process_ex) > 0:
            files = [item for item in files if item in only_process_ex]

    # make the tables from the scripts
    make_tables()

    # get a dictionary with example information from the example files
    ex_dict = get_examples_dict(verbose=verbose)

    # make the summary table for the LaTeX document
    build_tex_tables(ex_dict)

    # make the markdown table for ReadtheDocs
    build_md_tables(ex_dict)
