// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';
import { BLOCK_MAP } from './blockMap.mjs';
import { VoxelMob } from './VoxelMob.mjs';
import { VoxelPlayer} from './VoxelPlayer.mjs';
import { VW_ITEM_MAP } from './model_luts.mjs';

let camera, scene, renderer, controls, plane;
let objects = [];

let origin_offset, y_offset;  // Scene needs to be recentered on 0,0

const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

// Define cube geometry and materials
const geo = new THREE.BoxGeometry( 50, 50, 50 );

// Define grass once
let item_data = VW_ITEM_MAP["grass"];
const loader = new THREE.TextureLoader();
let grassMaterial = [
    new THREE.MeshBasicMaterial({ 
        map: loader.load('./block_textures/'+item_data["sides"]), 
        color: item_data["color"],
        opacity: item_data["opacity"],
        side: THREE.DoubleSide }), //right side
    new THREE.MeshBasicMaterial({ 
        map: loader.load('./block_textures/'+item_data["sides"]), 
        color: item_data["color"], 
        opacity: item_data["opacity"], 
        side: THREE.DoubleSide }), //left side
    new THREE.MeshBasicMaterial({ 
        map: loader.load('./block_textures/'+item_data["top"]), 
        color: item_data["color"], 
        opacity: item_data["opacity"], 
        side: THREE.DoubleSide }), //top side
    new THREE.MeshBasicMaterial({ 
        map: loader.load('./block_textures/'+item_data["bottom"]), 
        color: item_data["color"], 
        opacity: item_data["opacity"], 
        side: THREE.DoubleSide }), //bottom side
    new THREE.MeshBasicMaterial({ 
        map: loader.load('./block_textures/'+item_data["sides"]), 
        color: item_data["color"], 
        opacity: item_data["opacity"], 
        side: THREE.DoubleSide }), //front side
    new THREE.MeshBasicMaterial({ 
        map: loader.load('./block_textures/'+item_data["sides"]), 
        color: item_data["color"], 
        opacity: item_data["opacity"], 
        side: THREE.DoubleSide }), //back side
];

// Pull scene paths from input
let gtScenePath = sessionStorage.getItem('gtScene');

fetch(gtScenePath)  // Load the first file and save json data to object
.then(response => {
    return response.json();
})
.then(json => {
    let gtscene = json[0];
    loadScene(gtscene);
    render();
    addEventListeners();
    var canvii = document.getElementsByTagName("canvas");
    Array.from(canvii).forEach((canv) => {
        canv.style.display = "inline";
        canv.style.margin = "auto";
    });
})


function loadScene(json) {

    let blocks;
    if (typeof(json.blocks) == "string") {
        blocks = json.blocks.replaceAll('(','[').replaceAll(')',']');
        blocks = JSON.parse(blocks);
    }
    else {blocks = json.blocks}
    
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 10000 );
    camera.position.set( 400, 640, 1040 );
    camera.lookAt( 0, 0, 0 );

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xf0f0f0 );

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    plane = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene.add( plane );
    objects.push( plane );

    // find origin offset so that scene is centerd on 0,0
    let Xs = blocks.map(function(x) { return x[0]; });
    origin_offset = Math.floor( (Math.max(...Xs) + Math.min(...Xs)) / 2)
    let Ys = blocks.map(function(y) { return y[1]; });
    y_offset = Math.floor( Math.min(...Ys) )

    // Populate legend and mask colors
    let color_idx = 47;  // Increment mask color coding with each tag
    let blockColorList = [];
    blocks.forEach((block) => {
        if (!blockColorList.includes(block[3])) { blockColorList.push(block[3]) };
    });
    if (json.inst_seg_tags) {
        json.inst_seg_tags.forEach(tag => {
            while (blockColorList.includes(color_idx)) { color_idx++ }; // avoid colors of other blocks
            // For each mask block location, find the corresponding block in the block list and set the color to a unique value
            tag.locs.forEach((loc) => {
                let match_block_idx = blocks.findIndex((block) => block[0] == loc[0] && block[1] == loc[1] && block[2] == loc[2] && block[3] != 46 && block[3] != 0);
                if (match_block_idx != -1) blocks[match_block_idx][3] = color_idx;
                // Holes need their own number so they don't get texture later
                match_block_idx = blocks.findIndex((block) => block[0] == loc[0] && block[1] == loc[1] && block[2] == loc[2] && (block[3] == 0 || block[3] == 46) );
                if (match_block_idx != -1) blocks[match_block_idx][3] = color_idx + 20;
                // If the block doesn't exist, they marked the air
                match_block_idx = blocks.findIndex((block) => block[0] == loc[0] && block[1] == loc[1] && block[2] == loc[2]);
                if (match_block_idx == -1) blocks.push([loc[0], loc[1], loc[2], (color_idx + 20)]);
            });
        });
    }

    // load scene
    let cubeMaterial;
    for (let i=0; i<blocks.length; i++) {
        if (blocks[i][3] === 0) {  // if it's a hole, don't add anything
            continue;
        }
        else if (blocks[i][3] === 46) {  // if it's the ground, plant some grass
            cubeMaterial = grassMaterial;
        }
        else if (blocks[i][3] < 65) {
            cubeMaterial = new THREE.MeshLambertMaterial({ 
                color: BLOCK_MAP[blocks[i][3]],
                map: new THREE.TextureLoader().load( 'square-outline-textured.png' )
            });
        }
        else {
            cubeMaterial = new THREE.MeshLambertMaterial({ 
                color: BLOCK_MAP[blocks[i][3]],
                opacity: 0.7,
                transparent: true
            });
        }
        const voxel = new THREE.Mesh( geo, cubeMaterial );
        voxel.position.set(((blocks[i][0] - origin_offset)*50)+25, ((blocks[i][1] - y_offset)*50)+25, ((blocks[i][2] - origin_offset)*50)+25);
        scene.add( voxel );
        objects.push( voxel );
    }

    let world = {
        THREE: THREE,
        scene: scene,
        render: render,
        camera: camera,
    };
    let opts;

    // Load avatars and look arrow
    if (json.avatarInfo && json.agentInfo) {
        let user_pos = json.avatarInfo.pos;
        let agent_pos = json.agentInfo.pos;
        let user_look = lookRadsToVec(json.avatarInfo.look);
        
        // add user and agent avatars
        opts = {
            GLTFLoader: GLTFLoader,
            name: "player",
            position: [((user_pos[0]- origin_offset)*50)+25, ((user_pos[1] - y_offset)*50), ((user_pos[2]- origin_offset)*50)+25],
        };
        VoxelPlayer.build(world, opts).then(
            function (player) {
                player.rotate(json.avatarInfo.look[0]);
                // add look direction
                const head_position = new THREE.Vector3(player.mesh.position.x, player.mesh.position.y + 50, player.mesh.position.z);
                scene.add( new THREE.ArrowHelper( user_look, head_position, 150, 0xff0000, 40, 20 ) );
            }
        );
        opts = {
            GLTFLoader: GLTFLoader,
            name: "agent",
            position: [((agent_pos[0]- origin_offset)*50)+25, ((agent_pos[1] - y_offset)*50), ((agent_pos[2]- origin_offset)*50)+25],
        };
        VoxelPlayer.build(world, opts);
    }

    // Load mobs
    if (json.mobs) {
        json.mobs.forEach((mob) => {
            opts = {
                GLTFLoader: GLTFLoader,
                name: mob.mobtype,
                position: [((mob.pose[0] - origin_offset)*50)+25, ((mob.pose[1] - y_offset)*50), ((mob.pose[2]- origin_offset)*50)+25],
            };
            VoxelMob.build(world, opts);
        });
    }
    
    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scene.add( ambientLight );
    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scene.add( directionalLight );

    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    let cont = document.getElementById("voxel_viewer");
    cont.appendChild( renderer.domElement );

    // controls
    controls = new OrbitControls( camera, renderer.domElement );
    controls.listenToKeyEvents( window );
    controls.addEventListener( 'change', render );

    controls.enableZoom = true;
    controls.zoomSpeed = 0.5;
    controls.minPolarAngle = minCameraPitch;
    controls.maxPolarAngle = maxCameraPitch;
}

function lookRadsToVec(raw_vals) {
    // Look direction comes in as [yaw, pitch] floats
    // referenced s.t. pos pitch is down, 0 yaw = pos-z, and pos yaw is CCW
    // Three.js has pos pitch up, 0 yaw = pos-x, and pos yaw is CW
    const look_angles = [(-1)*raw_vals[0], ((-1)*raw_vals[1]) + Math.PI/2]
    // Convert pitch and yaw radian values to a Vector3 look direction
    let look_dir = new THREE.Vector3( 
        Math.cos(look_angles[0]) * Math.cos(look_angles[1]),
        Math.sin(look_angles[1]),
        Math.sin(look_angles[0]) * Math.cos(look_angles[1])
        );
    look_dir.normalize();

    return look_dir;
}

function addEventListeners() {
    window.addEventListener( 'resize', onWindowResize );
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
    render();
}

function render() {
    renderer.render( scene, camera );
}