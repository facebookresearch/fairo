import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get actuator xml from MJCF")
    parser.add_argument("filename")
    args = parser.parse_args()

    pat = re.compile('^\s*<\s*joint\s*name\s*=\s*"([\w\s]+)".*/\s*>\s*$')
    motor_format = '\t<motor joint="{joint_name}" name="{act_name}"/>'
    print("<actuator>")
    with open(args.filename) as f:
        lines = f.readlines()
        for line in lines:
            if m := pat.match(line):
                joint_name = m.groups()[0]
                act_name = joint_name.replace("joint", "actuator")
                print(
                    motor_format.format(
                        act_name, joint_name=joint_name, act_name=act_name
                    )
                )
    print("</actuator>")
