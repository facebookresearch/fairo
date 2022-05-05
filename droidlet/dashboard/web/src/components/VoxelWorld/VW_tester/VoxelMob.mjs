// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {VW_MOB_MAP} from "./model_luts.mjs"

class VoxelMob {
    constructor (model, world, opts) {
        this.world = world;
        this.opts = opts;
        this.mobType = opts.name;
        this.mesh = model;
        this.position_offset = opts.position_offset;
    }

    move(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.mesh.position.x += xyz.x;
        this.mesh.position.y += xyz.y;
        this.mesh.position.z += xyz.z;
    }

    moveTo(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        xyz = applyOffset(xyz, this.position_offset);
        this.mesh.position.x = xyz.x;
        this.mesh.position.y = xyz.y;
        this.mesh.position.z = xyz.z;
    }

    static build (world, opts) {
        let mob_data = VW_MOB_MAP[opts.name];
        opts.scale = opts.scale || 1.0;
        opts.scale *= mob_data.default_scale;
        opts.rotation = opts.rotation || [0, 0, 0];
        opts.rotation = applyOffset(opts.rotation, mob_data.rotation_offset)
        opts.position = opts.position || [0, 0, 0];
        opts.position = applyOffset(opts.position, mob_data.position_offset)
        opts.position_offset = mob_data.position_offset;

        const path = "./models/" + mob_data.model_folder;
        const loader = new opts.GLTFLoader();
        loader.setPath(path);
        return loader.loadAsync( mob_data.model_file ).then(
            function (gltf) {
                let model = gltf.scene;
                model.scale.multiplyScalar(opts.scale);
                model.position.set(opts.position[0], opts.position[1], opts.position[2]);
                model.rotation.x += opts.rotation[0];
                model.rotation.y += opts.rotation[1];
                model.rotation.z += opts.rotation[2];
                world.scene.add( model );
                return model;
            }
        ).then(
            function (model) {
               return new VoxelMob(model, world, opts);
           }
        );
    }
};

function parseXYZ (x, y, z) {
    if (typeof x === 'object' && Array.isArray(x)) {
        return { x: x[0], y: x[1], z: x[2] };
    }
    else if (typeof x === 'object') {
        return { x: x.x || 0, y: x.y || 0, z: x.z || 0 };
    }
    return { x: Number(x), y: Number(y), z: Number(z) };
}

function applyOffset (pos, offset) {
    // adjusts the passed in position/rotation to center the model in a voxel upright
    return [(pos[0] + offset[0]), (pos[1] + offset[1]), (pos[2] + offset[2])]
}


export {VoxelMob};