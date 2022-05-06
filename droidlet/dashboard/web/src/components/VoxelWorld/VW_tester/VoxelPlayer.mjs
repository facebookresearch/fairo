// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {VW_AVATAR_MAP} from "./model_luts.mjs"

class VoxelPlayer {
    constructor (model, world, opts) {
        this.world = world;
        this.opts = opts;
        this.avatarType = opts.name;
        this.mesh = model;
        this.position_offset = opts.position_offset;
        this.possessed = false;
        this.pov = 3;
        this.yaw = opts.rotation_offset[1];
        this.pitch = 0;  // skipping for now, hopefully won't need an offset
    };

    move(x, y, z) {
        this.mesh.position.x += x;
        this.mesh.position.y += y;
        this.mesh.position.z += z;
        if (this.possessed) this.updateCamera();
    };

    rotate(d_yaw, d_pitch) {
        // units of delta radians
        this.yaw += d_yaw;
        this.pitch += d_pitch;

        const euler = new this.world.THREE.Euler(
            Math.abs(Math.cos(this.yaw)) * this.pitch,
            this.yaw,
            Math.cos(Math.PI/2 + this.yaw) * this.pitch,
            'ZYX'
        );
        this.mesh.setRotationFromEuler(euler);
        if (this.possessed) this.updateCamera();
    };

    moveTo(x, y, z) {
        let xyz = applyOffset([x,y,z], this.position_offset);
        this.mesh.position.set(xyz[0], xyz[1], xyz[2]);
        if (this.possessed) this.updateCamera();
    };

    updatePov(type) {
        if (type === 'first' || type === 1) this.pov = 1;
        else if (type === 'third' || type === 3) this.pov = 3;
        this.possess();
    };

    toggle() {
        this.updatePov(this.pov === 1 ? 3 : 1);
        if (this.possessed) this.updateCamera();
    };

    updateCamera() {
        let cam_vector, final_cam_vector;

        let matrix = new this.world.THREE.Matrix4();
        matrix.extractRotation( this.mesh.matrix );

        if (this.pov === 1) {
            cam_vector = new this.world.THREE.Vector3( 50, 75, 0 );
            final_cam_vector = cam_vector.applyMatrix4( matrix );
            this.world.camera.position.copy( this.mesh.position ).add( final_cam_vector );
            this.world.camera.setRotationFromQuaternion(this.mesh.quaternion);
        } else {
            cam_vector = new this.world.THREE.Vector3( 0, 800, -800 );
            final_cam_vector = cam_vector.applyMatrix4( matrix );
            this.world.camera.position.copy( this.mesh.position ).add( final_cam_vector );
            this.world.camera.lookAt(this.mesh.position);
        }

        this.world.render();
    };

    possess() {
        if (!this.possessed) {
            this.possessed = true;
            this.updateCamera();
            return 1
        } else {
            return 0
        }
    };

    depossess() {
        if (this.possessed) {
            this.possessed = false;
            return 1
        } else {
            return 0
        }
    };

    static build (world, opts) {
        let avatar_data = VW_AVATAR_MAP[opts.name];
        opts.scale = opts.scale || 1.0;
        opts.scale *= avatar_data.default_scale;
        opts.rotation = opts.rotation || [0, 0, 0];
        opts.rotation = applyOffset(opts.rotation, avatar_data.rotation_offset)
        opts.rotation_offset = avatar_data.rotation_offset;
        opts.position = opts.position || [0, 0, 0];
        opts.position = applyOffset(opts.position, avatar_data.position_offset)
        opts.position_offset = avatar_data.position_offset;

        const path = "./models/" + avatar_data.model_folder;
        const loader = new opts.GLTFLoader();
        loader.setPath(path);
        return loader.loadAsync( avatar_data.model_file ).then(
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
               return new VoxelPlayer(model, world, opts);
           }
        );
    };

};

function applyOffset (pos, offset) {
    // adjusts the passed in position/rotation to center the model in a voxel upright
    return [(pos[0] + offset[0]), (pos[1] + offset[1]), (pos[2] + offset[2])]
}


export {VoxelPlayer};