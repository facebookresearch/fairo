// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { VoxelPlayer} from './VoxelPlayer.mjs';
import { VoxelItem} from './VoxelItem.mjs';
import { VoxelMob} from './VoxelMob.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';
import { VW_ITEM_MAP, VW_MOB_MAP, VW_AVATAR_MAP, MINECRAFT_BLOCK_MAP } from './model_luts.mjs'
import {traceRay} from './dvoxel_raycast.mjs'

const defaultCameraWidth = 512
const defaultCameraHeight = 512
const defaultCameraAspectRatio = defaultCameraWidth / defaultCameraHeight
const defaultCameraFOV = 45
const defaultCameraNearPlane = 1
const defaultCameraFarPlane = 10000
const fps = 2
const renderInterval = 1000 / fps
let camera, reticle, scene, renderer, loader, preLoadBlockMaterials, sceneItems;
const followPointerScale = 150;

const preLoadMaterialNames = ['grass', 'dirt', 'wood']//, 'white wool', 'orange wool', 'magenta wool'];
const blockScale = 50;
const bid2Name = {
    8: 'grass',
    9: 'dirt',
    13: 'wood',
    46: 'white wool',
    47: 'orange wool',
    48: 'magenta wool',
    49: 'light blue wool',
    50: 'yellow wool',
    51: 'lime wool',
    52: 'pink wool',
    53: 'gray wool',
    54: 'light gray wool',
    55: 'cyan wool',
    56: 'purple wool',
    57: 'blue wool',
    58: 'brown wool',
    59: 'green wool',
    60: 'red wool',
    61: 'black wool'
}

const TEXTURE_PATH = "https://cdn.jsdelivr.net/gh/snyxan/assets@main/block_textures/";


const SL = 16
const voxelOffset = [0, 0, 0]//[SL/2, SL/2, SL/2]

// voxel related constants
let voxels = new Array();
for (let ix = 0; ix < SL; ix++) {
    voxels[ix] = new Array();
    for (let iy = 0; iy < SL; iy++) {
        voxels[ix][iy] = new Array();
        for (let iz = 0; iz < SL; iz++) {
            voxels[ix][iy][iz] = 0;
        }
    }
}


const MoveStep = 0.5 // normalized -- block length is 1 here

let controlled_player, agent_player;
const AGENT_NAME = "craftassist_agent";
const PLAYER_NAME = "dashboard";

let mobs = {}
let mobList = []
let itemStacks = {}
let itemList = []

let direction_vec = new THREE.Vector3();


function pos2Name(x, y, z, box=false) {
    if (box) {
        return x + ',' + y + ',' + z + 'box'
    }
    return x + ',' + y + ',' + z
}


function handleKeypress(e, player) {
    let camera_vec;
    console.log(e.key)
    switch (e.key) {
        case "ArrowLeft":
            player.rotate(0.1);
            break;
        case "ArrowRight":
            player.rotate(-0.1);
            break;
        case "t":
            player.toggle();
            break;
        case "r":
            player.rotateTo(0,0);
            break;
        case "w":
            camera_vec = cameraVector();
            direction_vec.set(camera_vec[0], 0, camera_vec[2])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            player.move(direction_vec.x, direction_vec.y, direction_vec.z);
            updatePlayerPosition(player);
            break;
        case "s":
            camera_vec = cameraVector();
            direction_vec.set(-camera_vec[0], 0, -camera_vec[2])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            player.move(direction_vec.x, direction_vec.y, direction_vec.z);
            updatePlayerPosition(player);
            break;
        case "a":
            camera_vec = cameraVector();
            direction_vec.set(camera_vec[2], 0, -camera_vec[0])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            player.move(direction_vec.x, direction_vec.y, direction_vec.z);
            updatePlayerPosition(player);
            break;   
        case "d":
            camera_vec = cameraVector();
            direction_vec.set(-camera_vec[2], 0, camera_vec[0])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            player.move(direction_vec.x, direction_vec.y, direction_vec.z);
            updatePlayerPosition(player);
            break;    
        case "Shift":
            player.move(0, -1 * MoveStep * blockScale, 0);
            updatePlayerPosition(player);
            break;    
        case " ":
            player.move(0, 1 * MoveStep * blockScale, 0);
            updatePlayerPosition(player);
            break;    
    }
}


function applyCameraToPlayer(player) {
    player.possess();
    window.addEventListener("keydown", function (e) {
        handleKeypress(e, player);
    });
    document.addEventListener("mousemove", function (ev) {
      if (document.pointerLockElement === document.body) {
        let Xdiff = -ev.movementX / followPointerScale;
        let Ydiff = -ev.movementY / followPointerScale;
        
        player.cameraPitch(Ydiff);
        player.rotate(Xdiff);
        updatePlayerLook(player);
      }
    });
};

function updatePlayerLook(player) {
    let pitchYaw = player.getLookPitchYaw();
    let pitch = pitchYaw[0];
    let yaw = pitchYaw[1]
    let payload = {
        "status": "set_look",
        "pitch": pitch,
        "yaw": yaw
    }
    window.postMessage(payload, "*");
}

function updatePlayerPosition(player) {
    let pos = player.getPosition();
    let xyz = convertCoordinateSystems(
        pos.x / blockScale,
        pos.y / blockScale,
        pos.z / blockScale
    )
    let payload = {
        "status": "abs_move",
        "x": xyz[0],
        "y": xyz[1],
        "z": xyz[2]
    }
    window.postMessage(payload, "*");
}

function convertCoordinateSystems(x, y, z) {
    return [-x, y, z]
}


class DVoxelEngine {
    render() {
        this.renderer.render(this.scene, this.camera);
    }
    constructor (opts) {

        this.initTime = Date.now();

        this.scene = new THREE.Scene();
        scene = this.scene
        this.scene.background = new THREE.Color( 0xf0f0f0 );

        this.sceneItems = [];
        sceneItems = this.sceneItems;

        this.cameraWidth = opts.cameraWidth || defaultCameraWidth;
        this.cameraHeight = opts.cameraHeight || defaultCameraHeight;
        this.cameraAspectRatio = opts.cameraAspectRatio || defaultCameraAspectRatio;
        this.cameraFOV = opts.cameraFOV || defaultCameraFOV;
        this.cameraNearPlane = opts.cameraNearPlane || defaultCameraNearPlane;
        this.cameraFarPlane = opts.cameraFarPlane || defaultCameraFarPlane;

        this.camera = new THREE.PerspectiveCamera( this.cameraFOV, this.cameraWidth / this.cameraHeight, this.cameraNearPlane, this.cameraFarPlane);
        this.camera.position.set( 500, 800, 1300 );
        this.camera.lookAt( 0, 0, 0 );
        camera = this.camera;

        const reticleMaterial = new THREE.LineBasicMaterial({
            color: 0xff0000,
            linecap: "square"
        });
        
        let points = [
            new THREE.Vector3( -3, -3, 0 ),
            new THREE.Vector3( 0, 0, 0 ),
            new THREE.Vector3( 3, -3, 0 )
        ];
        const reticleGeo = new THREE.BufferGeometry().setFromPoints( points );
        reticle = new THREE.Line( reticleGeo, reticleMaterial );
        reticle.position.copy( camera.position );
        reticle.rotation.copy( camera.rotation );
        scene.add(reticle);

        const ambientLight = new THREE.AmbientLight( 0x606060 );
        this.scene.add( ambientLight );
        const directionalLight = new THREE.DirectionalLight( 0xffffff );
        directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
        this.scene.add( directionalLight );

        this.renderer = new THREE.WebGLRenderer( { antialias: true } );
        renderer = this.renderer
        this.renderer.setPixelRatio( window.devicePixelRatio );
        this.renderer.setSize( window.innerWidth, window.innerHeight );

        // Axis helper for debugging
        // this.scene.add( new THREE.AxesHelper( 10000 ) );

        // loader and preloaded materials -- to improve performance
        loader = new THREE.TextureLoader();
        preLoadBlockMaterials = new Map();
        preLoadMaterialNames.forEach(function (key, index) {
                let block_data = VW_ITEM_MAP[key];
                preLoadBlockMaterials.set(
                    key,
                    [
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"]}), //right side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"]}), //left side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["top"]), 
                            color: block_data["color"]}), //top side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["bottom"]), 
                            color: block_data["color"]}), //bottom side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"]}), //front side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"]}), //back side
                    ]);
            }
        );

        let world = {
            THREE: THREE,
            scene: scene,
            render: render,
            camera: camera,
            reticle: reticle,
            sceneItems: sceneItems,
        };

        for (const key in VW_AVATAR_MAP) {
            if (typeof(key) === "string" && VW_AVATAR_MAP[key] !== null) {
                const opts = {
                    GLTFLoader: GLTFLoader,
                    name: key,
                    position: [100, 500, -500]
                };
                VoxelPlayer.build(world, opts).then(
                    function (player) {
                        if (player.avatarType === "player") {
                            controlled_player = player;
                            applyCameraToPlayer(player);
                        }
                        if (player.avatarType === "agent") {
                            agent_player = player;
                        }
                    }
                );
            }
        };
        
        window.setInterval(render, renderInterval);
    }

    appendTo(element) {
        // console.log(element)
        element.appendChild(this.renderer.domElement)
    }


    setVoxel(pos, bid) {
        if (bid === 0) {
            let obj = scene.getObjectByName(pos2Name(pos[0], pos[1], pos[2]))
            console.log('deleting')
            console.log(obj)
            this.scene.remove(scene.getObjectByName(pos2Name(pos[0], pos[1], pos[2])))
            this.scene.remove(scene.getObjectByName(pos2Name(pos[0], pos[1], pos[2], true)))
            return;
        }
        const blockName = bid2Name[bid];
        const geometry = new THREE.BoxGeometry( blockScale, blockScale, blockScale );
        let block_data = VW_ITEM_MAP[blockName];
        let blockMaterials;
        if (preLoadBlockMaterials.has(blockName)) {
            // console.log('preloaddding!!!' + blockName)
            blockMaterials = preLoadBlockMaterials.get(blockName);
        } else {
            blockMaterials = [
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"]}), //right side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"]}), //left side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["top"]), 
                    color: block_data["color"]}), //top side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["bottom"]), 
                    color: block_data["color"]}), //bottom side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"]}), //front side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"]}), //back side
            ];
        }
        
        const cube = new THREE.Mesh(geometry, blockMaterials);
        cube.matrixAutoUpdate = false;
        cube.position.set(pos[0] * blockScale, pos[1] * blockScale, pos[2] * blockScale);
        cube.updateMatrix();
        cube.name = pos2Name(pos[0], pos[1], pos[2]);
        // console.log("Adding voxel with name: " + cube.name)
        this.scene.add(cube);
        this.sceneItems.push(cube);
        
        const box = new THREE.BoxHelper(cube, 0x000000);
        box.name = pos2Name(pos[0], pos[1], pos[2], true);
        this.scene.add(box);

        const bidx = convertCoordinateSystems(pos[0], pos[1], pos[2]);
        setBlock2(bidx[0], bidx[1], bidx[2], bid);
    }

    raycastVoxels(v) {
        let hitPosition = [0, 0, 0];
        let hitNormal = [0, 0, 0];
        let epsilon = 1e-8
        traceRay(v, cameraPosition(), cameraVector(), 30, hitPosition, hitNormal, epsilon)
        console.log('raycast result')
        console.log(hitPosition)
        return hitPosition;
    }

    getBlock(x, y, z) {
        // outside zone, always return 0 -- hack for raycasting
        if (x < 0 || x >= SL || y < 0 || y >= SL || z < 0 || z >= SL) {
            console.log("OUTSIDE RAYCAST REGION")
            return 0
        }
        console.log(x + ' ' + y + ' ' + z)
        return getBlock2(x, y, z)
    }

    updateAgents(agentsInfo) {
        // console.log("DVoxel Engine update agents")
        // console.log(agentsInfo);

        let that = this;
        agentsInfo.forEach(function(key, index) {
            let name = key["name"]
            let xyz = convertCoordinateSystems(
                key["x"],
                key["y"],
                key["z"]
            );
            let look = [key["yaw"], key["pitch"]];

            // console.log("name: " + name + "x: " + xyz[0] + ", y:" + xyz[1] + ", z:" + xyz[2])
            if (name === AGENT_NAME && agent_player != null) {
                agent_player.moveTo(xyz[0] * blockScale, xyz[1] * blockScale, xyz[2] * blockScale);
                agent_player.rotateTo(degToRad(look[0]), degToRad(look[1]));
                that.playerPostionSafetyCheck(agent_player);
            } else if (name === PLAYER_NAME && controlled_player != null) {
                // console.log("player moveTo: x: " + xyz[0] + ", y:" + xyz[1] + ", z:" + xyz[2]);
                controlled_player.moveTo(xyz[0] * blockScale, xyz[1] * blockScale, xyz[2] * blockScale);
                // controlled_player.rotateTo(degToRad(look[0]), degToRad(look[1]));
                // ^this is unstable, probably best to let the player object own look direction always
                that.playerPostionSafetyCheck(controlled_player);
            }
        })
    }

    playerPostionSafetyCheck(player) {
        if ((Date.now() - this.initTime) < 5000) {
            // Give everything time to load before panicking
            return
        }
        let pos = player.getPosition();
        let pos_xyz = convertCoordinateSystems(pos.x, pos.y, pos.z);
        if (((pos_xyz[0] / blockScale) > SL) ||
            ((pos_xyz[0] / blockScale) < 0) ||
            ((pos_xyz[1] / blockScale) > SL) ||
            ((pos_xyz[1] / blockScale) < 0) ||
            ((pos_xyz[2] / blockScale) > SL) ||
            ((pos_xyz[2] / blockScale) < 0) ) {
                console.log("safety fail, running away");
                // TODO check collisions and move somewhere else
                let safe_xyz = convertCoordinateSystems(1, SL-2, 1);
                player.moveTo(safe_xyz[0] * blockScale, safe_xyz[1] * blockScale, safe_xyz[2] * blockScale);
                updatePlayerPosition(player);
        }
    }

    updateMobs(mobsInfo) {
        // console.log("DVoxel Engine update mobs")
        // console.log(mobsInfo)
        let world = {
            THREE: THREE,
            scene: scene,
            render: render,
            camera: camera,
        };
        mobsInfo.forEach(function(key, index) {
            const entityId = key['entityId'].toString()
            const pos = convertCoordinateSystems(
                key['pos'][0],
                key['pos'][1],
                key['pos'][2]
            )
            const name = key['name']
            if (entityId in mobs) {
                // console.log("mob already exists, updating states")
                mobs[entityId].moveTo(pos[0] * blockScale, pos[1] * blockScale, pos[2] * blockScale);
                mobs[entityId].rotateTo(key['look'][0], key['look'][1]);
            } else if (mobList.includes(entityId)) {
                console.log("mob build race condition");
                // Mob still being built, ignore
            } else {
                console.log("building mob with ID: " + entityId);
                mobList.push(entityId);
                const mobOpts = {
                    GLTFLoader: GLTFLoader,
                    name: name,
                    position: [pos[0] * blockScale, pos[1] * blockScale, pos[2] * blockScale]
                };
                VoxelMob.build(world, mobOpts).then(
                    function (newMob) {
                        mobs[entityId] = newMob;
                        sceneItems.push(newMob.mesh);
                    }
                );
            }       
        })
    }

    updateItemStacks(itemStacksInfo) {
        // console.log("DVoxel Engine update item stacks")
        // console.log(itemStacksInfo)
        let world = {
            THREE: THREE,
            scene: scene,
            render: render,
            camera: camera,
        };
        itemStacksInfo.forEach(function(key, index) {
            const entityId = key['entityId'].toString()
            const pos = convertCoordinateSystems(
                key['pos'][0],
                key['pos'][1],
                key['pos'][2]
            )
            const name = key['typeName'];
            if (entityId in itemStacks) {
                // console.log("item already exists, updating states")
                itemStacks[entityId].moveTo(pos[0] * blockScale, pos[1] * blockScale, pos[2] * blockScale);
            } else if (itemList.includes(entityId)) {
                console.log("item build race condition");
                // Item still being built, ignore
            } else {
                console.log("building item with ID: " + entityId);
                itemList.push(entityId);
                const itemStackOpts = {
                    GLTFLoader: GLTFLoader,
                    name: name,
                    position: [pos[0] * blockScale, pos[1] * blockScale, pos[2] * blockScale]
                };
                VoxelItem.build(world, itemStackOpts).then(
                    function (newItemStack) {
                        itemStacks[entityId] = newItemStack;
                        sceneItems.push(newItemStack.mesh);
                    }
                );
            }            
        })
    }

    updateBlocks(blocksInfo) {
        // console.log("blocksInfo")
        // console.log(blocksInfo)
        let that = this
        blocksInfo.forEach(function(key, index) {
            let xyz = convertCoordinateSystems(
                key[0][0],
                key[0][1],
                key[0][2]
            )
            let idm = key[1]
            let bid = MINECRAFT_BLOCK_MAP[idm[0].toString() + "," + idm[1].toString()]
            // console.log("xyz: " + xyz + "  bid: " + bid)
            that.setVoxel([xyz[0],xyz[1],xyz[2]], bid);
        });
        // console.log("DVoxel Engine update blocks")
    }

    setBlock(x, y, z, idm) {
        // console.log("DVoxel Engine set block")
    }

    flashBlocks(bbox) {
        console.log("DVoxel Engine flash bbox")

        const coords = bbox.split(' ');
        const lowCorner = convertCoordinateSystems(parseInt(coords[0]), parseInt(coords[1]), parseInt(coords[2]));
        const highCorner = convertCoordinateSystems(parseInt(coords[3]), parseInt(coords[4]), parseInt(coords[5]));
        const geometry = new THREE.BoxGeometry(
            Math.abs((highCorner[0] - lowCorner[0]) * blockScale + 2),
            Math.abs((highCorner[1] - lowCorner[1]) * blockScale + 2),
            Math.abs((highCorner[2] - lowCorner[2]) * blockScale + 2),
        );
        const highlighterMaterial = new THREE.MeshBasicMaterial({color: 0x049ef4})
        const highlightCube = new THREE.Mesh(geometry, highlighterMaterial);
        highlightCube.position.x += (((highCorner[0] - lowCorner[0]) / 2) + lowCorner[0]) * blockScale;
        highlightCube.position.y += (((highCorner[1] - lowCorner[1]) / 2) + lowCorner[1]) * blockScale;
        highlightCube.position.z += (((highCorner[2] - lowCorner[2]) / 2) + lowCorner[2]) * blockScale;
        scene.add(highlightCube);

        let flashInterval = window.setInterval(function () {
            if (highlightCube.visible) {
                highlightCube.visible = false;
            } else {
                highlightCube.visible = true;
            }
            render();
        }, 500);

        window.setTimeout(function () {
            window.clearInterval(flashInterval);
            scene.remove(highlightCube);
        }, 5100);

    }

}

function degToRad(deg) {
    return (deg / 360) * Math.PI * 2
}

function setBlock2(x, y, z, id) {
    voxels[x + voxelOffset[0]][y + voxelOffset[1]][z + voxelOffset[2]] = id
}

function getBlock2(x, y, z) {
    if (x + voxelOffset[0] < 0 || x + voxelOffset[0] >= SL || y + voxelOffset[1] < 0 || y + voxelOffset[1] >= SL || z + voxelOffset[2] < 0 || z + voxelOffset[2] >= SL) {
        console.log("Get Block 2 out of index")
    }
    return voxels[x + voxelOffset[0]][y + voxelOffset[1]][z + voxelOffset[2]]
}

function cameraVector() {
    let temporaryVector = new THREE.Vector3();
    camera.getWorldDirection(temporaryVector);
    return [temporaryVector.x, temporaryVector.y, temporaryVector.z];
  }

function cameraPosition() {
    let temporaryPosition = new THREE.Vector3;
    temporaryPosition.multiplyScalar(0)
    temporaryPosition.applyMatrix4(camera.matrixWorld)
    return [temporaryPosition.x / blockScale, temporaryPosition.y / blockScale, temporaryPosition.z / blockScale]
  }

function render() {
    renderer.render( scene, camera );
}


export {DVoxelEngine, SL};