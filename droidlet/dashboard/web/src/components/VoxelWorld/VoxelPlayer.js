var skin = require('minecraft-skin');


class VoxelPlayer {
    constructor (world, img, skinOpts) {
        this.possessed;
    
        skinOpts.scale = skinOpts.scale || new world.THREE.Vector3(0.04, 0.04, 0.04);  // Check scale assumptions
        this.playerSkin = new skin(world.THREE, img, skinOpts);
        this.player = this.playerSkin.mesh;  // Returns player object
        
        this.player.position.set(0, 562, -20);  // Check starting position assumptions
        world.scene.add(this.player);
        
        world.control(this.player);  // To think about more
        
        this.pov = 1;   
    }

    move(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.player.position.x += xyz.x;
        this.player.position.y += xyz.y;
        this.player.position.z += xyz.z;
    }

    moveTo(x, y, z) {
        var xyz = parseXYZ(x, y, z);
        this.player.position.x = xyz.x;
        this.player.position.y = xyz.y;
        this.player.position.z = xyz.z;
    }

    updatePov(type) {
        if (type === 'first' || type === 1) {
            this.pov = 1;
        }
        else if (type === 'third' || type === 3) {
            this.pov = 3;
        }
        this.possess();
    }

    toggle() {
        this.updatePov(this.pov === 1 ? 3 : 1);
    }

    possess() {  
        if (this.possessed) this.possessed.remove(world.camera);
        var key = this.pov === 1 ? 'cameraInside' : 'cameraOutside';
        this.player[key].add(world.camera);
        this.possessed = this.player[key];
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

export default VoxelPlayer;