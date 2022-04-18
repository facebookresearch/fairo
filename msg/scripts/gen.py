import dataclasses
import os
import glob
import re
import subprocess


@dataclasses.dataclass
class RosMsgFile:
    path: str
    namespace: str
    name: str


def to_camel_case(token):
    return token[0].lower() + token.replace("_", " ").title().replace(" ", "")[1:]


def rosmsg_files():
    paths = glob.glob(f"{os.environ['CONDA_PREFIX']}/share/*_msgs/msg/*.msg")
    for path in paths:
        match = re.match(".*/share/(.*)_msgs/msg/(.*).msg", path)
        namespace, name = match.groups()
        yield RosMsgFile(path, namespace, name)


def rosfile2capstruct(rosmsg_file):
    special_case = {
        ("std", "Time"): (
            set(),
            [
                "  sec @0 :UInt32;",
                "  nsec @1 :UInt32;",
            ],
        ),
        ("std", "Duration"): (
            set(),
            [
                "  sec @0 :Int32;",
                "  nsec @1 :Int32;",
            ],
        ),
    }
    if (rosmsg_file.namespace, rosmsg_file.name) in special_case:
        return special_case[(rosmsg_file.namespace, rosmsg_file.name)]

    primitive_type_map = {
        "bool": "Bool",
        "byte": "UInt8",
        "char": "Int8",
        "uint8": "UInt8",
        "uint16": "UInt16",
        "uint32": "UInt32",
        "uint64": "UInt64",
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "float32": "Float32",
        "float64": "Float64",
        "string": "Text",
        "time": "std_msgs/Time",
        "duration": "std_msgs/Duration",
    }

    if rosmsg_file.namespace == "std" and rosmsg_file.name in primitive_type_map.values():
        return None

    import_set = set()
    struct_fields = []

    field_id = 0
    for line in open(rosmsg_file.path):
        if "#" in line:
            line = line.split("#")[0]

        line = line.strip()
        if not line:
            continue

        if match := re.fullmatch("([\w//]+)(\[\d*\])?\W+(\w+)", line):
            field_type, is_arr, field_name = match.groups()
            field_name = to_camel_case(field_name)

            field_type = primitive_type_map.get(field_type, field_type)

            if "/" in field_type:
                namespace, field_type = field_type.split("/")
                if namespace != f"{rosmsg_file.namespace}_msgs":
                    import_set.add(
                        f'using {field_type} = import "{namespace}.capnp".{field_type};'
                    )

            if field_type == "Header" and rosmsg_file.namespace != "std":
                import_set.add('using Header = import "std_msgs.capnp".Header;')

            if is_arr:
                field_type = f"List({field_type})"

            if field_type in ["List(UInt8)", "List(Int8)"]:
                field_type = "Data"

            struct_fields.append(f"  {field_name} @{field_id} :{field_type};")
            field_id += 1
        elif match := re.fullmatch("(\w+)\W+(\w+)\W*=\W*(\w+)", line):
            const_type, const_name, const_value = match.groups()

            const_type = primitive_type_map.get(const_type, const_type)
            const_name = to_camel_case("k_" + const_name)

            struct_fields.append(f"  const {const_name} :{const_type} = {const_value};")
        else:
            assert False

    return import_set, struct_fields


new_files = {}
for i, rosmsg_file in enumerate(rosmsg_files()):
    if rosmsg_file.namespace not in new_files:
        new_files[rosmsg_file.namespace] = {"import_set": set(), "content": []}
    capstruct = rosfile2capstruct(rosmsg_file)
    if not capstruct:
        continue
    import_set, struct_fields = capstruct
    file = new_files[rosmsg_file.namespace]
    file["import_set"] |= import_set
    file["content"].append(f"struct {rosmsg_file.name} {{")
    file["content"].extend(struct_fields)
    file["content"].append("}")


out_dir = "src/fairomsg/def"
os.makedirs(out_dir, exist_ok=True)
for filename, info in new_files.items():
    print(filename)
    with open(os.path.join(out_dir, f"{filename}_msgs.capnp"), "w") as out:
        print(subprocess.check_output(["capnp", "id"]).decode().strip() + ";", file=out)
        print('using Cxx = import "/capnp/c++.capnp";', file=out)
        print(f'$Cxx.namespace("mrp::{filename}");', file=out)

        for import_item in sorted(info["import_set"]):
            print(import_item, file=out)

        for line in info["content"]:
            print(line, file=out)
