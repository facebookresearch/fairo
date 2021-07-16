# Kuka data files

We use [xacro](http://wiki.ros.org/xacro) to generate our [urdf](http://wiki.ros.org/urdf)s to reduce the number of copied XML snippets.

There are two ways to use the xacro files:

- To generate urdf files onto the filesystem, run `./generate_urdf.sh`.
- To dynamically generate urdf files at runtime, specify `xacro` file with the correct parameters is enough for `RobotInterface`.

## Adding modifications

1. Modify `./xacro/iiwa7.urdf.xacro` to include the parameter
2. Modify `./xacro/iiwa7.xacro` to include a [conditional block](http://wiki.ros.org/xacro#Conditional_Blocks) for your macro.
3. Add your macro to a xacro file under `./xacro/`.
4. Add a xacro command with the right parameters under `generate_urdf.sh`.
