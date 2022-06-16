// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {MathUtils} from './three.module.mjs';
import {VW_AVATAR_MAP} from "./model_luts.mjs"
const PLAYER_PATH = "https://cdn.jsdelivr.net/gh/snyxan/assets@main/models/";

class VoxelPlayer {
    constructor (model, world, opts) {
        this.world = world;
        this.opts = opts;
        this.avatarType = opts.name;
        this.scale = opts.scale;
        this.mesh = model;
        this.position_offset = opts.position_offset;
        this.rotation_offset = opts.rotation_offset;
        this.possessed = false;
        this.pov = 3;
        this.matrix = new world.THREE.Matrix4();
        this.cam_vector = new world.THREE.Vector3();
        this.cam_pitch = 0;
        this.visible = true;
    };

    move(x, y, z) {
        console.log(x + ","+y+","+z)
        this.mesh.position.x += x;
        this.mesh.position.y += y;
        this.mesh.position.z += z;
        console.log(this.mesh.position)
        if (this.possessed) this.updateCamera();
    };

    moveTo(x, y, z) {
        let xyz = applyOffset([x,y,z], this.position_offset);
        this.mesh.position.set(xyz[0], xyz[1], xyz[2]);
        if (this.possessed) this.updateCamera();
    };

    rotate(d_yaw) {
        this.mesh.rotateY(d_yaw);
        if (this.possessed) this.updateCamera();
    };

    rotateTo(yaw, pitch) {
        this.mesh.rotation.set(this.opts.rotation_offset[0], this.opts.rotation_offset[1], this.opts.rotation_offset[2])
        this.mesh.rotateY(yaw)
        this.cam_pitch = 0
        this.cameraPitch(pitch)
        if (this.possessed) this.updateCamera();
    }

    getPitchYaw() {
        let pitch = MathUtils.radToDeg(this.mesh.rotation.x);
        let yaw = MathUtils.radToDeg(this.mesh.rotation.y);
        return [pitch, yaw]
    }

    getPosition() {
        return this.mesh.position;
    }

    updatePov(type) {
        if (type === 'first' || type === 1) this.pov = 1;
        else if (type === 'third' || type === 3) this.pov = 3;
        this.cam_pitch = 0;
        this.possess();
    }

    toggle() {
        this.updatePov(this.pov === 1 ? 3 : 1);
        if (this.possessed) this.updateCamera();
    }

    remove() {
        if (this.visible) {
            if (this.possessed) this.depossess();
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

    cameraPitch(d_pitch) {
        this.cam_pitch += d_pitch;
    };

    updateCamera() {
        let final_cam_vector;
        
        // Different offsets may break this
        this.matrix.extractRotation( this.mesh.matrix );

        if (this.pov === 1) {
            this.cam_vector.set(0, (40*this.scale), (50*this.scale));
            final_cam_vector = this.cam_vector.applyMatrix4( this.matrix );
            this.world.camera.position.copy( this.mesh.position ).add( final_cam_vector );
            this.world.camera.setRotationFromEuler( this.mesh.rotation );
            this.world.camera.rotation.x += this.opts.rotation_offset[0];
            this.world.camera.rotation.y += this.opts.rotation_offset[1] + Math.PI;
            // ^This seems like a hack, maybe come back to. Why does it look towards the mesh?
            this.world.camera.rotation.z += this.opts.rotation_offset[2];
            this.world.camera.rotateX(this.cam_pitch);
        } else {
            this.cam_vector.set( 0, (700*this.scale), (-900*this.scale) );
            final_cam_vector = this.cam_vector.applyMatrix4( this.matrix );
            this.world.camera.position.copy( this.mesh.position ).add( final_cam_vector );
            this.world.camera.lookAt( this.mesh.position );
            this.world.camera.rotateX(this.cam_pitch);
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
        avatar_data.default_scale *= opts.scale;
        opts.rotation = opts.rotation || [0, 0, 0];
        opts.rotation = applyOffset(opts.rotation, avatar_data.rotation_offset)
        opts.rotation_offset = avatar_data.rotation_offset;
        opts.position = opts.position || [0, 0, 0];
        opts.position = applyOffset(opts.position, avatar_data.position_offset)
        opts.position_offset = avatar_data.position_offset;

        const path = PLAYER_PATH + avatar_data.model_folder;
        const loader = new opts.GLTFLoader();
        loader.setPath(path);
        return loader.loadAsync( avatar_data.model_file ).then(
            function (gltf) {
                let model = gltf.scene;
                model.scale.multiplyScalar(avatar_data.default_scale);
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