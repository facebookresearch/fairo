
class VoxelMob {
    constructor (world, opts) {
        this.world = world;
        this.opts = opts;
        // this.opts.position = this.opts.position || [0, 0, 0];  // Check starting position assumptions
        }

    move(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.mob.position.x += xyz.x;
        this.mob.position.y += xyz.y;
        this.mob.position.z += xyz.z;
    }

    moveTo(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.mob.position.x = xyz.x;
        this.mob.position.y = xyz.y;
        this.mob.position.z = xyz.z;
    }
};

class ChickenMob extends VoxelMob {
    constructor (model, world, opts) {
        super(world, opts);
        this.mob = model;
    }

    static build (world, opts) {
        opts.scale = opts.scale || 90.0;  // adjusted to be ~1 voxel in size
        opts.rotation = opts.rotation || [0, 0, 0];  // model rotation OK
        opts.position = opts.position || [0, 0, 0];
        opts.position = positionOffset(opts.position, [11, 0, 23])  // Move to middle of voxel

        const path = "./chicken_model/";
        const modelFile = "chicken.gltf";
        const loader = new opts.GLTFLoader();
        loader.setPath(path);
        return loader.loadAsync( modelFile ).then(
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
               return new ChickenMob(model, world, opts);
           }
        );
    }
}

function parseXYZ (x, y, z) {
    if (typeof x === 'object' && Array.isArray(x)) {
        return { x: x[0], y: x[1], z: x[2] };
    }
    else if (typeof x === 'object') {
        return { x: x.x || 0, y: x.y || 0, z: x.z || 0 };
    }
    return { x: Number(x), y: Number(y), z: Number(z) };
}

function positionOffset (pos, offset) {
    // adjusts the passed in position to center the model in a voxel
    return [(pos[0] + offset[0]), (pos[1] + offset[1]), (pos[2] + offset[2])]
}


export {ChickenMob};