// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { VoxelPlayer} from './VoxelPlayer.mjs';
import { VoxelItem} from './VoxelItem.mjs';
import { VoxelMob} from './VoxelMob.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';
import { VW_ITEM_MAP, VW_MOB_MAP, VW_AVATAR_MAP } from './model_luts.mjs'
import { OrbitControls } from './OrbitControls.mjs';
import {traceRay} from './dvoxel_raycast.mjs'

const defaultCameraWidth = 512
const defaultCameraHeight = 512
const defaultCameraAspectRatio = defaultCameraWidth / defaultCameraHeight
const defaultCameraFOV = 45
const defaultCameraNearPlane = 1
const defaultCameraFarPlane = 10000
const fps = 20
const renderInterval = 1000 / fps
let controls, camera, scene, renderer, loader, preLoadBlockMaterials

const preLoadMaterialNames = ['grass', 'dirt']//, 'white wool', 'orange wool', 'magenta wool'];
const blockScale = 50;
const bid2Color = {
    1: 0x808080,
    2: 0xff0000,
    3: 0xffff00,
    4: 0x800000,
    5: 0x0000ff
}
const bid2Name = {
    8: 'grass',
    9: 'dirt',
    46: 'white wool',
    47: 'orange wool',
    48: 'magenta wool'
}

const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

const TEXTURE_PATH = "https://cdn.jsdelivr.net/gh/snyxan/assets@main/block_textures/";


const SL = 16
const voxelOffset = [SL/2, SL/2, SL/2]

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

let controlled_player;


function pos2Name(x, y, z, box=false) {
    if (box) {
        return x + ',' + y + ',' + z + 'box'
    }
    return x + ',' + y + ',' + z
}



function walkabout(obj, dist, world) {
    let dir = Math.floor(3 * Math.random());
    let choices = [-1, 1];
    let move = choices[Math.floor(choices.length * Math.random())] * dist;
    switch (dir) {
        case 0:
            if (obj.mesh.position.x < 500 && obj.mesh.position.x > -500){
                obj.move(move, 0, 0);
            } else {
                obj.moveTo(0,0,0);
            }
            break;
        case 1:
            // obj.move(0, move, 0);
            break;
        case 2:
            if (obj.mesh.position.z < 500 && obj.mesh.position.z > -500){
                obj.move(0, 0, move);
            } else {
                obj.moveTo(0,0,0);
            }
            break;
    }
    render();
}
function handleKeypress(e, player) {
    let camera_vec, direction_vec, control_pos
    console.log(e.key)
    switch (e.key) {
        case "ArrowLeft":
            player.rotate(0.1, 0);
            break;
        case "ArrowRight":
            player.rotate(-0.1, 0);
            break;
        case "ArrowUp":
            player.rotate(0, 0.1);
            break;
        case "ArrowDown":
            player.rotate(0, -0.1);
            break;
        case "t":
            player.toggle();
            break;
        case "w":
            camera_vec = cameraVector();
            direction_vec = new THREE.Vector3(camera_vec[0], 0, camera_vec[2])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            control_pos = player.mesh.position;
            player.move(direction_vec.x, direction_vec.y, direction_vec.z)
            break;
        case "s":
            camera_vec = cameraVector();
            direction_vec = new THREE.Vector3(-camera_vec[0], 0, -camera_vec[2])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            control_pos = player.mesh.position;
            player.move(direction_vec.x, direction_vec.y, direction_vec.z)
            break;
        case "a":
            camera_vec = cameraVector();
            direction_vec = new THREE.Vector3(camera_vec[2], 0, camera_vec[0])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            control_pos = player.mesh.position;
            player.move(direction_vec.x, direction_vec.y, direction_vec.z)
            break;   
        case "d":
            camera_vec = cameraVector();
            direction_vec = new THREE.Vector3(-camera_vec[2], 0, -camera_vec[0])
            direction_vec.normalize()
            direction_vec.multiplyScalar(MoveStep * blockScale)
            control_pos = player.mesh.position;
            player.move(direction_vec.x, direction_vec.y, direction_vec.z)
            break;    
        case "Shift":
            player.move(0, -1 * MoveStep * blockScale, 0)
            break;    
        case " ":
            player.move(0, 1 * MoveStep * blockScale, 0)
            break;    
    }
}


function cameraTest(player) {
    controls.enabled = false;
    window.addEventListener("keydown", function (e) {
        handleKeypress(e, player);
    });
    player.possess();
};


class DVoxelEngine {
    render() {
        this.renderer.render(this.scene, this.camera);
    }
    constructor (opts) {

        this.scene = new THREE.Scene();
        scene = this.scene
        this.scene.background = new THREE.Color( 0xf0f0f0 );


        this.cameraWidth = opts.cameraWidth || defaultCameraWidth;
        this.cameraHeight = opts.cameraHeight || defaultCameraHeight;
        this.cameraAspectRatio = opts.cameraAspectRatio || defaultCameraAspectRatio;
        this.cameraFOV = opts.cameraFOV || defaultCameraFOV;
        this.cameraNearPlane = opts.cameraNearPlane || defaultCameraNearPlane;
        this.cameraFarPlane = opts.cameraFarPlane || defaultCameraFarPlane;

        this.camera = new THREE.PerspectiveCamera( this.cameraFOV, this.cameraWidth / this.cameraHeight, this.cameraNearPlane, this.cameraFarPlane);
        camera = this.camera
        this.camera.position.set( 500, 800, 1300 );
        this.camera.position.set( 1000, 1600, 2600 );
        this.camera.lookAt( 0, 0, 0 );

        const gridHelper = new THREE.GridHelper( 1000, 20 );
        this.scene.add( gridHelper );
        const geometry = new THREE.PlaneGeometry( 1000, 1000 );
        geometry.rotateX( - Math.PI / 2 );
        let plane = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
        this.scene.add( plane );

        const ambientLight = new THREE.AmbientLight( 0x606060 );
        this.scene.add( ambientLight );
        const directionalLight = new THREE.DirectionalLight( 0xffffff );
        directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
        this.scene.add( directionalLight );

        this.renderer = new THREE.WebGLRenderer( { antialias: true } );
        renderer = this.renderer
        this.renderer.setPixelRatio( window.devicePixelRatio );
        this.renderer.setSize( window.innerWidth, window.innerHeight );

        controls = new OrbitControls( this.camera, this.renderer.domElement );
        controls.listenToKeyEvents( window );
        controls.addEventListener( 'change', render );

        controls.enableZoom = true;
        controls.zoomSpeed = 0.5;
        controls.minPolarAngle = minCameraPitch;
        controls.maxPolarAngle = maxCameraPitch;


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
                            color: block_data["color"],
                            opacity: block_data["opacity"],
                            transparent: true,
                            side: THREE.DoubleSide }), //right side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"], 
                            opacity: block_data["opacity"], 
                            transparent: true, 
                            side: THREE.DoubleSide }), //left side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["top"]), 
                            color: block_data["color"], 
                            opacity: block_data["opacity"], 
                            transparent: true, 
                            side: THREE.DoubleSide }), //top side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["bottom"]), 
                            color: block_data["color"], 
                            opacity: block_data["opacity"], 
                            transparent: true, 
                            side: THREE.DoubleSide }), //bottom side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"], 
                            opacity: block_data["opacity"], 
                            transparent: true, 
                            side: THREE.DoubleSide }), //front side
                        new THREE.MeshBasicMaterial({ 
                            map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                            color: block_data["color"], 
                            opacity: block_data["opacity"], 
                            transparent: true, 
                            side: THREE.DoubleSide }), //back side
                    ]);
            }
        );
        // for (const key in preLoadMaterialNames) {
            
        // }

        let world = {
            THREE: THREE,
            scene: scene,
            render: render,
            camera: camera,
        };

        for (const key in VW_AVATAR_MAP) {
            if (typeof(key) === "string" && VW_AVATAR_MAP[key] !== null) {
                const opts = {
                    GLTFLoader: GLTFLoader,
                    name: key,
                    position: [50, 250, 50]
                };
                VoxelPlayer.build(world, opts).then(
                    function (player) {
                        // window.setInterval(walkabout, 1000, player, 50, world);
                        if (player.avatarType === "player") {
                            controlled_player = player;
                            cameraTest(player);
                        }
                    }
                );
            }
        };
        
        const TEST_ITEMS = ['pink wool', 'white wool', 'blue wool', 'brown wool', 'grass']
        TEST_ITEMS.forEach(function(key, index) {
            console.log(key)
            let max = 5, min = -5
            let ix = Math.floor(Math.random() * (max - min + 1) + min)
            let iz = Math.floor(Math.random() * (max - min + 1) + min)
            const itemOpts = {
                GLTFLoader: GLTFLoader,
                name: key,
                position: [ix * blockScale, 5 * blockScale, iz * blockScale]
            };
            VoxelItem.build(world, itemOpts).then(
                function (item) {
                    // window.setInterval(walkabout, 1000, item, 50);
                }
            );
            }
        ) 

        const TEST_MOBS = ['cow', 'chicken']
        TEST_MOBS.forEach(function(key, index) {
            console.log(key)
            let max = 5, min = -5
            let ix = Math.floor(Math.random() * (max - min + 1) + min)
            let iz = Math.floor(Math.random() * (max - min + 1) + min)
            const mobOpts = {
                GLTFLoader: GLTFLoader,
                name: key,
                position: [ix * blockScale, 5 * blockScale, iz * blockScale]
            };
            VoxelMob.build(world, mobOpts).then(
                function (mob) {
                    window.setInterval(walkabout, 1000, mob, 50);
                }
            );
            }
        ) 
        
        
        window.setInterval(render, renderInterval);
    }

    appendTo(element) {
        console.log(element)
        element.appendChild(this.renderer.domElement)
    }


    setVoxel(pos, bid) {
        if (bid == 0) {
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
            console.log('preloaddding!!!' + blockName)
            blockMaterials = preLoadBlockMaterials.get(blockName);
        } else {
            blockMaterials = [
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"],
                    opacity: block_data["opacity"],
                    transparent: true,
                    side: THREE.DoubleSide }), //right side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"], 
                    opacity: block_data["opacity"], 
                    transparent: true, 
                    side: THREE.DoubleSide }), //left side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["top"]), 
                    color: block_data["color"], 
                    opacity: block_data["opacity"], 
                    transparent: true, 
                    side: THREE.DoubleSide }), //top side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["bottom"]), 
                    color: block_data["color"], 
                    opacity: block_data["opacity"], 
                    transparent: true, 
                    side: THREE.DoubleSide }), //bottom side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"], 
                    opacity: block_data["opacity"], 
                    transparent: true, 
                    side: THREE.DoubleSide }), //front side
                new THREE.MeshBasicMaterial({ 
                    map: loader.load(TEXTURE_PATH+block_data["sides"]), 
                    color: block_data["color"], 
                    opacity: block_data["opacity"], 
                    transparent: true, 
                    side: THREE.DoubleSide }), //back side
            ];
        }
        
        // const material = new THREE.MeshBasicMaterial( {color: colorCode} );
        // const cube = new THREE.Mesh( geometry, material );
        const cube = new THREE.Mesh( geometry, blockMaterials );
        cube.position.set(pos[0] * blockScale, pos[1] * blockScale, pos[2] * blockScale)
        // const cubeAABB = cube.geometry.computeBoundingBox();
        cube.name = pos2Name(pos[0], pos[1], pos[2])
        console.log("Adding voxel with name: " + cube.name)
        this.scene.add( cube );
        const box = new THREE.BoxHelper(cube, 0x000000);
        box.name = pos2Name(pos[0], pos[1], pos[2], true)
        this.scene.add(box);

        setBlock2(pos[0], pos[1], pos[2], bid)
        // const cubeAABB = new THREE.Mesh( cube, new THREE.MeshBasicMaterial( 0xff0000 ) );
        // cubeAABB.geometry.computeBoundingBox();
        // const box = new THREE.Box3();
        // box.copy( cube.geometry.boundingBox ).applyMatrix4( cube.matrixWorld );
        // const box = new THREE.Box3().setFromObject( cubeAABB, 0xffff00 );
        // this.scene.add( box );
        // const geometry2 = new THREE.BoxGeometry( 50, 50, 50 );
        // const material2 = new THREE.MeshBasicMaterial( {color: 0xff0000} );
        // const cube2 = new THREE.Mesh( geometry2, material2 );
        // cube2.position.set(150, 75, 100)
        // this.scene.add( cube2 );
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
        if (x < -SL/2 || x >= SL/2 || y < -SL/2 || y >= SL/2 || z < -SL/2 || z >= SL/2) {
            console.log("OUTSIDE RAYCAST REGION")
            return 0
        }
        console.log(x + ' ' + y + ' ' + z)
        return getBlock2(x, y, z)
    }
}

function setBlock2(x, y, z, id) {
    voxels[x + voxelOffset[0]][y + voxelOffset[1]][z + voxelOffset[2]] = id
}

function getBlock2(x, y, z) {
    console.log
    return voxels[x + voxelOffset[0]][y + voxelOffset[1]][z + voxelOffset[2]]
}

function cameraVector() {
    let temporaryVector = new THREE.Vector3;
    temporaryVector.multiplyScalar(0)
    temporaryVector.z = -1
    temporaryVector.transformDirection( camera.matrixWorld )
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