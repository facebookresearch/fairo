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
        this.hoverID = null;
    }

    move(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.mesh.position.x += xyz.x;
        this.mesh.position.y += xyz.y;
        this.mesh.position.z += xyz.z;
    };

    moveTo(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        xyz = applyOffset(xyz, [25,25,25]);
        this.mesh.position.x = xyz.x;
        this.mesh.position.y = xyz.y;
        this.mesh.position.z = xyz.z;
    };

    drop() {
        if (this.location === "inventory") {
            this.world.scene.add(this.mesh);
            this.location = "world";
            this.hoverID = window.setInterval(hover, 100, this);
            // FIXME need to update position to be near the avatar doing the dropping
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

        let item_data = VW_ITEM_MAP[opts.name];
        let itemMaterial;
        if (typeof(item_data) === "number") {
            itemMaterial = new world.THREE.MeshBasicMaterial( { color: item_data, opacity: 1.0 } );
        }
        const geo = new world.THREE.BoxGeometry( 20, 20, 20 );
        let itemMesh = new world.THREE.Mesh( geo, itemMaterial );
        opts.position = opts.position || [0, 0, 0];
        opts.position = applyOffset(opts.position, [25,25,25]);  // move to the center of the voxel
        itemMesh.position.set(opts.position[0], opts.position[1], opts.position[2]);
        itemMesh.rotation.set(0, Math.PI/4, Math.PI/4);
        world.scene.add(itemMesh);

        return new Promise(resolve => {
            const item = new VoxelItem(itemMesh, world, opts);
            opts.hoverID = window.setInterval(hover, 100, item);
            item.hoverID = opts.hoverID;
            resolve(item);
        });
    };
};

function hover(obj) {
    // This can't possibly be the right way to do this...
    let vel;
    let rel_pos = obj.mesh.position.y % 50;
    if (rel_pos <= 30) {
        vel = (rel_pos - 14) / 2;
    } else {
        vel = Math.abs(rel_pos - 46) / 2;
    }
    obj.mesh.position.y += (vel * obj.hoverDirection);
    obj.mesh.rotation.y += 0.05;

    rel_pos += (vel * obj.hoverDirection);
    if (rel_pos < 15 || rel_pos > 45) {
        obj.hoverDirection *= -1;
    }

    obj.world.render();
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


export {VoxelItem};