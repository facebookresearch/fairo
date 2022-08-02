// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {VW_ITEM_MAP} from "./model_luts.mjs"

const MODEL_PATH = "https://cdn.jsdelivr.net/gh/snyxan/assets@main/models/";

class VoxelItem {
    constructor (model, world, opts) {
        this.world = world;
        this.opts = opts;
        this.scale = opts.scale;
        this.location = "world";
        this.itemType = opts.name;
        this.mesh = model;
        this.hoverDirection = 1;
        this.hoverID;
    }

    move(x, y, z) {
        this.mesh.position.x += x;
        this.mesh.position.y += y;
        this.mesh.position.z += z;
    };

    moveTo(x, y, z) {
        if (x != this.mesh.position.x || z != this.mesh.position.z) {
            // only update if x or z positions have changed, otherwise it breaks float
            this.mesh.position.set(x, y, z);
        }
    };

    drop() {
        if (this.location === "inventory") {
            this.world.scene.add(this.mesh);
            this.location = "world";
            this.hoverID = window.setInterval(hover, 100, this);
            // FIXME item moveTo voxel in front of avatar
            return 1
        } else {
            return 0
        };
    };

    pick() {
        if (this.location === "world") {
            this.world.scene.remove(this.mesh);
            this.location = "inventory";
            clearInterval(this.hoverID);
            return 1
        } else {
            return 0
        };
    };

    static build (world, opts) {
        // Load item model or textures
        let loader, item;

        let item_data = VW_ITEM_MAP[opts.name];
        opts.scale = opts.scale || 1.0;
        opts.position = opts.position || [0, 0, 0];
        opts.position_offset = item_data.position_offset || [(25*opts.scale),(25*opts.scale),(25*opts.scale)];
        opts.position = applyOffset(opts.position, opts.position_offset);
        opts.rotation = opts.rotation || [0, 0, 0];
        opts.rotation_offset = item_data.rotation_offset || [0, Math.PI/4, Math.PI/4];
        opts.rotation = applyOffset(opts.rotation, opts.rotation_offset);

        if ("model_file" in item_data) {
            // This is a GTFL model

            const path = MODEL_PATH + item_data.model_folder;
            loader = new opts.GLTFLoader();
            loader.setPath(path);

            return loader.loadAsync( item_data.model_file ).then(
                function (gltf) {
                    let model = gltf.scene;
                    model.scale.multiplyScalar(opts.scale * item_data.default_scale);
                    model.position.set(opts.position[0], opts.position[1], opts.position[2]);
                    model.rotation.x += opts.rotation[0];
                    model.rotation.y += opts.rotation[1];
                    model.rotation.z += opts.rotation[2];
                    world.scene.add( model );
                    return model;
                }
            ).then(
                function (model) {
                    item = new VoxelItem(model, world, opts);
                    item.hoverID = window.setInterval(hover, 150, item);
                    return item
               }
            );

        } else {
            // This is a material, load the texture

            loader = new world.THREE.TextureLoader();
            const itemMaterials = [
                new world.THREE.MeshBasicMaterial({ 
                    map: loader.load('./block_textures/'+item_data["sides"]), 
                    color: item_data["color"] }), //right side
                new world.THREE.MeshBasicMaterial({ 
                    map: loader.load('./block_textures/'+item_data["sides"]), 
                    color: item_data["color"] }), //left side
                new world.THREE.MeshBasicMaterial({ 
                    map: loader.load('./block_textures/'+item_data["top"]), 
                    color: item_data["color"] }), //top side
                new world.THREE.MeshBasicMaterial({ 
                    map: loader.load('./block_textures/'+item_data["bottom"]), 
                    color: item_data["color"] }), //bottom side
                new world.THREE.MeshBasicMaterial({ 
                    map: loader.load('./block_textures/'+item_data["sides"]), 
                    color: item_data["color"] }), //front side
                new world.THREE.MeshBasicMaterial({ 
                    map: loader.load('./block_textures/'+item_data["sides"]), 
                    color: item_data["color"] }), //back side
            ];
            const geo = new world.THREE.BoxGeometry( (20*opts.scale), (20*opts.scale), (20*opts.scale) );
            let itemMesh = new world.THREE.Mesh( geo, itemMaterials );
            itemMesh.position.set(opts.position[0], opts.position[1], opts.position[2]);
            itemMesh.rotation.set(opts.rotation[0], opts.rotation[1], opts.rotation[2]);

            world.scene.add(itemMesh);

            return new Promise(resolve => {
                item = new VoxelItem(itemMesh, world, opts);
                item.hoverID = window.setInterval(hover, 150, item);
                resolve(item);
            });
        }
    };
};

function hover(obj) {
    let vel;
    let rel_pos = obj.mesh.position.y % (50*obj.scale);
    if (rel_pos <= (30 * obj.scale)) {
        vel = Math.abs(rel_pos - (14*obj.scale)) / 2;
    } else {
        vel = Math.abs(rel_pos - (46*obj.scale)) / 2;
    }
    vel += 0.1;
    obj.mesh.position.y += (vel * obj.hoverDirection * obj.scale);
    obj.mesh.rotation.y += 0.05;

    rel_pos += (vel * obj.hoverDirection * obj.scale);
    if (rel_pos < (15*obj.scale)) {
        obj.hoverDirection = 1;
    } else if (rel_pos > (45*obj.scale)) {
        obj.hoverDirection = -1;
    }

    obj.world.render();
};

function applyOffset (pos, offset) {
    // adjusts the passed in position/rotation to center the model in a voxel upright
    return [(pos[0] + offset[0]), (pos[1] + offset[1]), (pos[2] + offset[2])]
}


export {VoxelItem};