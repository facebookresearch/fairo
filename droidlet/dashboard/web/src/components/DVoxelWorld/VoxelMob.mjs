// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {VW_MOB_MAP} from "./model_luts.mjs"

const MOB_PATH = "https://cdn.jsdelivr.net/gh/snyxan/assets@main/models/";

class VoxelMob {
    constructor (model, world, opts) {
        this.world = world;
        this.opts = opts;
        this.mobType = opts.name;
        this.mesh = model;
        this.position_offset = opts.position_offset;
        this.rotation_offset = opts.rotation_offset;
        this.visible = true;
        this.rotation = {"pitch": 0, "yaw": 0};
        this.worldX = new world.THREE.Vector3(1, 0, 0);
        this.worldY = new world.THREE.Vector3(0, 1, 0);
    }

    move(x, y, z) {
        this.mesh.position.x += x;
        this.mesh.position.y += y;
        this.mesh.position.z += z;
    }

    moveTo(x, y, z) {
        let xyz = applyOffset([x,y,z], this.position_offset);
        this.mesh.position.set(xyz[0], xyz[1], xyz[2]);
    }

    rotateTo(yaw, pitch) {
        if (this.rotation.yaw != yaw || this.rotation.pitch != pitch) {
            this.mesh.rotation.set(this.rotation_offset[0], this.rotation_offset[1], this.rotation_offset[2]);
            this.mesh.rotateOnWorldAxis(this.worldX, pitch);
            this.mesh.rotateOnWorldAxis(this.worldY, yaw);
            this.rotation.yaw = yaw;
            this.rotation.pitch = pitch;
        }
    }

    remove() {
        if (this.visible) {
            this.world.scene.remove(this.mesh);
            this.visible = false;
        }
    }

    add() {
        if (!this.visible) {
            this.world.scene.add(this.mesh);
            this.visible = true;
        }
    }

    static build (world, opts) {
        let mob_data = VW_MOB_MAP[opts.name];
        opts.scale = opts.scale || 1.0;
        opts.scale *= mob_data.default_scale;
        opts.rotation = opts.rotation || [0, 0, 0];
        opts.rotation = applyOffset(opts.rotation, mob_data.rotation_offset)
        opts.position = opts.position || [0, 0, 0];
        opts.position = applyOffset(opts.position, mob_data.position_offset)
        opts.rotation_offset = mob_data.rotation_offset;
        opts.position_offset = mob_data.position_offset;

        const path = MOB_PATH + mob_data.model_folder;
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

function applyOffset (pos, offset) {
    // adjusts the passed in position/rotation to center the model in a voxel upright
    return [(pos[0] + offset[0]), (pos[1] + offset[1]), (pos[2] + offset[2])]
}


export {VoxelMob};