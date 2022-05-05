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
        this.mob = model;
    }

    move(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.mob.position.x += xyz.x;
        this.mob.position.y += xyz.y;
        this.mob.position.z += xyz.z;
    };

    moveTo(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.mob.position.x = xyz.x;
        this.mob.position.y = xyz.y;
        this.mob.position.z = xyz.z;
    };

    drop() {
        if (this.location === "inventory") {
            this.world.scene.add(this.mob);
            this.location = "world";
            return 1
        } else {
            return 0
        };
    };

    pick() {
        if (this.location === "world") {
            this.world.scene.remove(this.mob);
            this.location = "inventory";
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
        opts.position = opts.position.map(x => x+25); // move to the center of the voxel
        itemMesh.position.set(opts.position[0], opts.position[1], opts.position[2]);
        itemMesh.rotation.set(0, Math.PI/4, Math.PI/4);
        world.scene.add(itemMesh);

        return new Promise(resolve => {
              resolve(new VoxelItem(itemMesh, world, opts));
        });
    };
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


export {VoxelItem};