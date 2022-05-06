// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import {VW_ITEM_MAP} from "./model_luts.mjs"

class VoxelItem {
    constructor (model, world, opts) {
        this.world = world;
        this.opts = opts;
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
        let xyz = applyOffset([x,y,z], [25,25,25]);
        this.mesh.position.set(xyz[0], xyz[1], xyz[2]);
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
        // This could all live in the constructor, but leaving it this way for now to
        // 1) mirror the syntax of VoxelMob and 2) allow for easy extension to load models

        // Load texture images and apply to geometry
        let item_data = VW_ITEM_MAP[opts.name];
        const loader = new world.THREE.TextureLoader();
        const itemMaterials = [
            new world.THREE.MeshBasicMaterial({ 
                map: loader.load('./block_textures/'+item_data["sides"]), 
                color: item_data["color"],
                opacity: item_data["opacity"],
                transparent: true,
                side: world.THREE.DoubleSide }), //right side
            new world.THREE.MeshBasicMaterial({ 
                map: loader.load('./block_textures/'+item_data["sides"]), 
                color: item_data["color"], 
                opacity: item_data["opacity"], 
                transparent: true, 
                side: world.THREE.DoubleSide }), //left side
            new world.THREE.MeshBasicMaterial({ 
                map: loader.load('./block_textures/'+item_data["top"]), 
                color: item_data["color"], 
                opacity: item_data["opacity"], 
                transparent: true, 
                side: world.THREE.DoubleSide }), //top side
            new world.THREE.MeshBasicMaterial({ 
                map: loader.load('./block_textures/'+item_data["bottom"]), 
                color: item_data["color"], 
                opacity: item_data["opacity"], 
                transparent: true, 
                side: world.THREE.DoubleSide }), //bottom side
            new world.THREE.MeshBasicMaterial({ 
                map: loader.load('./block_textures/'+item_data["sides"]), 
                color: item_data["color"], 
                opacity: item_data["opacity"], 
                transparent: true, 
                side: world.THREE.DoubleSide }), //front side
            new world.THREE.MeshBasicMaterial({ 
                map: loader.load('./block_textures/'+item_data["sides"]), 
                color: item_data["color"], 
                opacity: item_data["opacity"], 
                transparent: true, 
                side: world.THREE.DoubleSide }), //back side
        ];
        opts.scale = opts.scale || 1.0;
        const geo = new world.THREE.BoxGeometry( (20*opts.scale), (20*opts.scale), (20*opts.scale) );
        let itemMesh = new world.THREE.Mesh( geo, itemMaterials );

        opts.position = opts.position || [0, 0, 0];
        opts.position = applyOffset(opts.position, [25,25,25]);  // move to the center of the voxel
        itemMesh.position.set(opts.position[0], opts.position[1], opts.position[2]);
        itemMesh.rotation.set(0, Math.PI/4, Math.PI/4);
        world.scene.add(itemMesh);

        return new Promise(resolve => {
            const item = new VoxelItem(itemMesh, world, opts);
            item.hoverID = window.setInterval(hover, 100, item);
            resolve(item);
        });
    };
};

function hover(obj) {
    // This can't possibly be the right way to do this...
    let vel;
    let rel_pos = obj.mesh.position.y % 50;
    if (rel_pos <= 30) {
        vel = Math.abs(rel_pos - 14) / 2;
    } else {
        vel = Math.abs(rel_pos - 46) / 2;
    }
    obj.mesh.position.y += (vel * obj.hoverDirection);
    obj.mesh.rotation.y += 0.05;

    rel_pos += (vel * obj.hoverDirection);
    if (rel_pos < 15) {
        obj.hoverDirection = 1;
    } else if (rel_pos > 45) {
        obj.hoverDirection = -1;
    }

    obj.world.render();
};

function applyOffset (pos, offset) {
    // adjusts the passed in position/rotation to center the model in a voxel upright
    return [(pos[0] + offset[0]), (pos[1] + offset[1]), (pos[2] + offset[2])]
}


export {VoxelItem};