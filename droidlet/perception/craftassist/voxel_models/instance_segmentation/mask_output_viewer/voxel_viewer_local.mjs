// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';
import { BLOCK_MAP } from './blockMap.mjs';

let cameras = {1: null, 2: null};
let controls = {1: null, 2: null};
let scenes = {1: null, 2: null};
let renderers = {1: null, 2: null};
let planes = {1: null, 2: null};
let objects = {1: [], 2: []};

let avatarsOff = false;

let origin_offset, y_offset;  // Scene needs to be recentered on 0,0

const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

// Define cube geometry and materials
const geo = new THREE.BoxGeometry( 50, 50, 50 );

// Pull scene paths from input
let modelOutputPath = 'scene_list.json';

let modelScene, numScenes;
let sceneIdx = 0;
fetch(modelOutputPath)  // Load the second file
.then(response => {
    return response.json();
})
.then(json => {
    modelScene = json;
    numScenes = modelScene.length;
    loadScene(modelScene[sceneIdx], 1);
    addEventListeners();
    render();
    var canvii = document.getElementsByTagName("canvas");
    Array.from(canvii).forEach((canv) => {
        canv.style.display = "inline";
        canv.style.margin = "auto";
    });
    // canvii[1].style.float = "right";
    return modelScene.length < numScenes ? modelScene.length : numScenes;
})
.then(sceneLimit => {
    document.getElementById("prevScene").addEventListener("click", function() {
        sceneIdx > 0 ? sceneIdx-- : sceneIdx = sceneLimit - 1;
        refreshCanvii();
    });
    document.getElementById("nextScene").addEventListener("click", function() {
        sceneIdx < (sceneLimit - 1) ? sceneIdx++ : sceneIdx = 0;
        refreshCanvii();
    });
})


function refreshCanvii() {
    var canvii = document.getElementsByTagName("canvas");
    Array.from(canvii).forEach((canv) => {
        canv.parentNode.removeChild(canv);
    });
    let obj = {msg: "clear"};
    window.parent.postMessage(JSON.stringify(obj), "*");

    objects = {1: [], 2: []};
    loadScene(modelScene[sceneIdx], 1);
    render();

    canvii = document.getElementsByTagName("canvas");
    Array.from(canvii).forEach((canv) => {
        canv.style.display = "inline";
        canv.style.margin = "auto";
    });
    // canvii[1].style.float = "right";
}


function loadScene(scene, idx) {

    let blocks;
    if (typeof(scene.blocks) == "string") {
        blocks = scene.blocks.replaceAll('(','[').replaceAll(')',']');
        blocks = JSON.parse(blocks);
    }
    else {blocks = scene.blocks}
    
    cameras[idx] = new THREE.PerspectiveCamera( 50, (window.innerWidth) / (window.innerHeight - 50), 1, 10000 );
    cameras[idx].position.set( 400, 640, 1040 );
    cameras[idx].lookAt( 0, 0, 0 );

    scenes[idx] = new THREE.Scene();
    scenes[idx].background = new THREE.Color( 0xf0f0f0 );

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    planes[idx] = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scenes[idx].add( planes[idx] );
    objects[idx].push( planes[idx] );

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
    if (scene.inst_seg_tags) {
        scene.inst_seg_tags.forEach(tag => {
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
            let obj = {msg: {idx: idx, label: tag, color: BLOCK_MAP[color_idx++]}};
            window.parent.postMessage(JSON.stringify(obj), "*");
        });
    }

    // load scene
    let cubeMaterial;
    for (let i=0; i<blocks.length; i++) {
        if (blocks[i][3] === 0) {  // if it's a hole, don't add anything
            continue;
        }
        else if (scene.blocks[i][3] === 46 || scene.blocks[i][3] === 9 || scene.blocks[i][3] === 8) {  // if it's the ground, skip the texture and add lines instead
            cubeMaterial = new THREE.MeshBasicMaterial( { color: 0xffffff, opacity: 0.3, transparent: true } );
            const edges = new THREE.EdgesGeometry( geo );  // outline the white blocks for visibility
            const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x000000 } ) );
            line.position.set(((blocks[i][0] - origin_offset)*50)+25, ((blocks[i][1] - y_offset)*50)+25, ((blocks[i][2] - origin_offset)*50)+25);
            scenes[idx].add( line );
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
        scenes[idx].add( voxel );
        objects[idx].push( voxel );
    }

    // Load avatars and look arrow
    if (scene.avatarInfo && scene.agentInfo && !avatarsOff) {
        let user_pos = scene.avatarInfo.pos;
        let agent_pos = scene.agentInfo.pos;
        let user_look = lookRadsToVec(scene.avatarInfo.look);
        const head_position = new THREE.Vector3((user_pos[0]*50)+25, (user_pos[1]*50)+100, (user_pos[2]*50)+25);
        const userMaterial = new THREE.MeshLambertMaterial( { color: 0xffff00 } );
        const agentMaterial = new THREE.MeshLambertMaterial( { color: 0x337eff } );

        // add look direction
        scenes[idx].add( new THREE.ArrowHelper( user_look, head_position, 150, 0xff0000, 40, 20 ) );

        // add user and agent avatars
        addAvatar(scenes[idx], user_pos, userMaterial, scene.avatarInfo.look);
        addAvatar(scenes[idx], agent_pos, agentMaterial, scene.agentInfo.look);
    }
    
    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scenes[idx].add( ambientLight );
    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scenes[idx].add( directionalLight );

    renderers[idx] = new THREE.WebGLRenderer( { antialias: true } );
    renderers[idx].setPixelRatio( window.devicePixelRatio );
    renderers[idx].setSize( (window.innerWidth), (window.innerHeight - 50) );
    let cont = document.getElementById("voxel_viewer");
    cont.appendChild( renderers[idx].domElement );

    // controls
    controls[idx] = new OrbitControls( cameras[idx], renderers[idx].domElement );
    controls[idx].listenToKeyEvents( window );
    controls[idx].addEventListener( 'change', render );

    controls[idx].enableZoom = true;
    controls[idx].zoomSpeed = 0.5;
    controls[idx].minPolarAngle = minCameraPitch;
    controls[idx].maxPolarAngle = maxCameraPitch;
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

// Add user or agent avatars to the scene
function addAvatar(scene, position, material, look_dir) {
    const loader = new GLTFLoader();
    loader.load( './robot.glb', function ( gltf ) {
        let model = gltf.scene;
        model.scale.multiplyScalar(25.0);
        model.position.set((position[0]*50)+25, (position[1]*50), (position[2]*50)+25)
        model.rotation.y += look_dir[0]; //yaw, referenced a la the raw vals for some reason
        scene.add( model );
        model.traverse( function ( object ) {
            if ( object.isMesh ) {
                object.castShadow = false;
                object.material = material;
            }
        } );
    } );
}

function addEventListeners() {
    document.addEventListener( 'pointermove', onPointerMove );
    window.addEventListener( 'resize', onWindowResize );
    window.addEventListener( 'wheel', onWheel );
}

function onWindowResize() {
    for (let idx in cameras) {
        if (cameras[idx]) {
            cameras[idx].aspect = (window.innerWidth) / (window.innerHeight - 50);
            cameras[idx].updateProjectionMatrix();
            renderers[idx].setSize( (window.innerWidth), (window.innerHeight - 50) );
        }
    }
    render();
}

function onWheel ( event ) {
    matchCameras();
    render();
}

function matchCameras() {
    // Set camera two to match camera one
    if (cameras[2]) {
        cameras[2].position.set( cameras[1].position.x, cameras[1].position.y, cameras[1].position.z );
        cameras[2].lookAt( 0, 0, 0 );
    }
}

function onPointerMove( event ) {
    matchCameras();
    render();
}

function render() {
    renderers[1].render( scenes[1], cameras[1] );
}