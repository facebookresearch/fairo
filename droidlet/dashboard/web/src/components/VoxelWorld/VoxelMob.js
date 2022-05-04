

class VoxelMob {
    constructor (world, opts) {
        this.opts = opts;
        this.opts.position = this.opts.position || [0, 562, -20];  // Check starting position assumptions
        this.mob = new world.THREE.Object3D();  // Placeholder object, implemented by subclass
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

function parseXYZ (x, y, z) {
    if (typeof x === 'object' && Array.isArray(x)) {
        return { x: x[0], y: x[1], z: x[2] };
    }
    else if (typeof x === 'object') {
        return { x: x.x || 0, y: x.y || 0, z: x.z || 0 };
    }
    return { x: Number(x), y: Number(y), z: Number(z) };
}

class ChickenMob extends VoxelMob {
    constructor (world, opts) {
        super(world, opts);
        this.opts.scale = this.opts.scale || 1.0  // adjust after test
        this.opts.rotation = this.opts.rotation || [0, 0, 0]  // adjust after test

        // load chicken gltf model
        // CC-NC license
        // https://sketchfab.com/3d-models/chicken-0098586123fc435f86c7f5cb3d3b8a79
        let path = "./chicken_model/";
        let model = "chicken.gltf";
        addModel(this.opts.GLTFLoader, world.scene, this.opts.scale, this.opts.position, this.opts.rotation, path, model);
    }
}

function addModel(GLTFLoader, scene, scale, position, rotation, path, model) {
    const loader = new GLTFLoader();
    loader.setPath(path);
    loader.load( model, function ( gltf ) {
        let model = gltf.scene;
        model.scale.multiplyScalar(scale);
        model.position.set(position[0], position[1], position[2])
        model.rotation.x += rotation[0];
        model.rotation.y += rotation[1];
        model.rotation.z += rotation[2];
        scene.add( model );
        // model.traverse( function ( object ) {
        //     if ( object.isMesh ) {
        //         object.castShadow = false;
        //     }
        // } );
    } );
}


export {ChickenMob};