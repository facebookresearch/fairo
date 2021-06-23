# VoxelWorld HOWTO

VoxelWorld is a self-contained html which is compiled separately from the main dashboard app and is used as an iframe in VoxelWorld component.

### How to modify&compile VoxelWorld

1. Make changes to VoxelWorld source code
2. Rename package.json.local to package.json (this is to avoid conflict with main dashboard app)
3. run `browserify api.js -s myWorld > ../../../public/VoxelWorld/myWorld.js` to compile source code, the newly compiled `myWorld.js` will be used as an iframe in VoxelWorld component.
4. Rename package.json back to package.json.local
