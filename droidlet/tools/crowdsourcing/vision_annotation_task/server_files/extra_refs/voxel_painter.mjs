// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';
import { staticShapes } from './staticShapes.mjs';
import { BLOCK_MAP } from './blockMap.mjs';

let cameras = {1: null, 2: null};
let controls = {1: null, 2: null};
let scenes = {1: null, 2: null};
let renderers = {1: null, 2: null};
let planes = {1: null, 2: null};
let pointers = {1: null, 2: null};
let raycasters = {1: null, 2: null};
let user_raycasters = {1: null, 2: null};
let objects = {1: [], 2: []};

let marked_blocks = [];
let actions_taken = []; // [original_block, new_block, action_type]
let inst_seg_tags = [{"tags": [], "locs": []}]

let isADown = false;
let isSDown = false;
let isDDown = false;
let startedHIT = false;
let isQual = false;
let avatarsOff = false;
let twoWindows = true;

let origin_offset;  // Scene needs to be recentered on 0,0 and then annotation output needs to be reindexed to the original origin reference

const addAxes = false;  // Useful for development. The positive X axis is red, Y is green, Z  is blue.
const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

// Define cube geometry and materials
const geo = new THREE.BoxGeometry( 50, 50, 50 );
const rollOverMaterial = new THREE.MeshBasicMaterial( { color: 0xff0000, opacity: 0.3, transparent: true } );
const cubeMaterial_mark = new THREE.MeshBasicMaterial( { color: 0x089000, opacity: 0.5, transparent: true } );
let rollOverMesh = new THREE.Mesh( geo, rollOverMaterial );

// Pull scene key from URL params
const urlParams = new URLSearchParams(window.location.search);
let module_key = "";
let scene_idx = "";
if (urlParams.get('batch_id')) {
    module_key = urlParams.get('batch_id');
    scene_idx = urlParams.get('scene_idx');
    inst_seg_tags[0]["tags"].push(urlParams.get('label'));
}
else if (urlParams.get('scene_filename')){
    module_key = urlParams.get('scene_filename');
    scene_idx = urlParams.get('scene_idx');
    avatarsOff = true;
    twoWindows = false;
}
console.log("Module key: " + module_key);

// Load the appropriate scene and number of windows based on module key
if (module_key.includes("test")){  // Used for loading test scenes
    isQual = true;
    let shapes = new staticShapes( module_key );
    if (!shapes) console.error("Scene for " + module_key + " did not load correctly");
    else init(shapes.scene, true);
}
else if (module_key[0] === 'q') {  // This is the qualification HIT, load the appropriate scene
    isQual = true;
    let shapes = new staticShapes( parseInt(module_key[1]) );
    if (!shapes) console.error("Scene for " + module_key + " did not load correctly");
    else init(shapes.scene, true);
}
else {  // Not a test or qual HIT, look for the scene list file
    fetch("./scene_list.json")
    .then(response => {
        return response.json();
    })
    .then(jsondata => {
        return jsondata[parseInt(scene_idx)];  // Pull the appropriate scene from the file
    })
    .then(scene => {
        init(scene);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}

function init(scene) {
    console.log("Initializing scene");
    loadScene(scene, 1);
    if (twoWindows) loadScene(scene, 2);
    addEventListeners();
    render();
    var canvii = document.getElementsByTagName("canvas");
    Array.from(canvii).forEach((canv) => {
        canv.style.display = "inline";
        canv.style.margin = "auto";
    });
    if (twoWindows) canvii[1].style.float = "right";
}

function loadScene(scene, idx) {
    
    cameras[idx] = new THREE.PerspectiveCamera( 50, (window.innerWidth/2) / (window.innerHeight - 50), 1, 10000 );
    cameras[idx].position.set( 400, 640, 1040 );
    cameras[idx].lookAt( 0, 0, 0 );

    scenes[idx] = new THREE.Scene();
    scenes[idx].background = new THREE.Color( 0xf0f0f0 );

    if (idx === 2) {
        // some things only for scene #2
        scenes[idx].add( rollOverMesh );

        // pointer for clicking
        raycasters[idx] = new THREE.Raycaster();
        pointers[idx] = new THREE.Vector2();
    }

    // Load the ground plane grid iff qual HIT
    if (isQual){
        const gridHelper = new THREE.GridHelper( 20*50 , 20 );
        scenes[idx].add( gridHelper );
    }

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    planes[idx] = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scenes[idx].add( planes[idx] );
    objects[idx].push( planes[idx] );

    // find origin offset so that scene is centerd on 0,0
    let Xs = scene.blocks.map(function(x) { return x[0]; });
    origin_offset = Math.floor( (Math.max(...Xs) + Math.min(...Xs)) / 2)

    // load scene
    let cubeMaterial;
    for (let i=0; i<scene.blocks.length; i++) {
        if (scene.blocks[i][3] === 0) {  // if it's a hole, don't add anything
            continue;
        }
        else if (scene.blocks[i][3] === 46 || scene.blocks[i][3] === 9 || scene.blocks[i][3] === 8) {  // if it's the ground, skip the texture and add lines instead
            cubeMaterial = new THREE.MeshBasicMaterial( { color: 0xffffff, opacity: 1.0 } );
            const edges = new THREE.EdgesGeometry( geo );  // outline the white blocks for visibility
            const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x000000 } ) );
            line.position.set(((scene.blocks[i][0] - origin_offset)*50)+25, (scene.blocks[i][1]*50)+25, ((scene.blocks[i][2] - origin_offset)*50)+25);
            scenes[idx].add( line );
        }
        else {
            cubeMaterial = new THREE.MeshLambertMaterial( { 
                color: BLOCK_MAP[scene.blocks[i][3]], 
                map: new THREE.TextureLoader().load( 'square-outline-textured.png' ) 
            });
        }
        const voxel = new THREE.Mesh( geo, cubeMaterial );
        voxel.position.set(((scene.blocks[i][0] - origin_offset)*50)+25, (scene.blocks[i][1]*50)+25, ((scene.blocks[i][2] - origin_offset)*50)+25);
        voxel.name = scene.blocks[i][0] + "_" + scene.blocks[i][1] + "_" + scene.blocks[i][2] + "_" + scene.blocks[i][3];
        scenes[idx].add( voxel );
        objects[idx].push( voxel );
    }

    // add axes helper if turned on
    if (addAxes) {
        scenes[idx].add( new THREE.AxesHelper( 10000 ) );
    }

    if (scene.avatarInfo && scene.agentInfo && !avatarsOff) {
        let user_pos = scene.avatarInfo.pos;
        let agent_pos = scene.agentInfo.pos;
        let user_look = lookRadsToVec(scene.avatarInfo.look);
        const head_position = new THREE.Vector3((user_pos[0]*50)+25, (user_pos[1]*50)+100, (user_pos[2]*50)+25);
        const userMaterial = new THREE.MeshLambertMaterial( { color: 0xffff00 } );
        const agentMaterial = new THREE.MeshLambertMaterial( { color: 0x337eff } );
        const targetGeo = new THREE.BoxGeometry( 0, 50, 50 );
        const targetMaterial = new THREE.MeshLambertMaterial( { color: 0xfeb74c, map: new THREE.TextureLoader().load( 'target.png' ), transparent: true } );

        // add look direction and init look raycaster - MUST BE AFTER SCENE, BEFORE USER
        if (!isQual) {  // Skip if it's a qualification HIT
            scenes[idx].updateMatrixWorld();  // Must call this for raycasting to work before render
            scenes[idx].add( new THREE.ArrowHelper( user_look, head_position, 150, 0xff0000, 40, 20 ) );
            user_raycasters[idx] = new THREE.Raycaster(head_position, user_look);
            const intersects = user_raycasters[idx].intersectObjects( objects[idx] );
            if ( intersects.length > 0 ) {
                const intersect = intersects[ 0 ];
                let target = new THREE.Mesh( targetGeo, targetMaterial );
                target.position.set(intersect.point.x, intersect.point.y, intersect.point.z);
                target.rotation.y += user_look.y;
                target.rotation.z -= user_look.z;
                scenes[idx].add(target);
            }
        }

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
    renderers[idx].setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    let cont = document.getElementById("voxel_painter");
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
    document.addEventListener( 'pointerdown', onPointerDown );
    document.addEventListener( 'keydown', onDocumentKeyDown );
    document.addEventListener( 'keyup', onDocumentKeyUp );
    window.addEventListener( 'resize', onWindowResize );
    window.addEventListener( 'wheel', onWheel );
}

function onWindowResize() {
    for (let idx in cameras) {
        cameras[idx].aspect = (window.innerWidth/2) / (window.innerHeight - 50);
        cameras[idx].updateProjectionMatrix();
        renderers[idx].setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
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
    if (pointers[2]) {
        pointers[2].set( ( (event.clientX - (window.innerWidth/1.9)) / (window.innerWidth/2.1) ) * 2 - 1, - ( event.clientY / (window.innerHeight - 50) ) * 2 + 1 );
        raycasters[2].setFromCamera( pointers[2], cameras[2] );
        // Select which blocks the raycaster should intersect with
        let objs_to_intersect = objects[2];
        if (isSDown) objs_to_intersect = objects[2].concat(marked_blocks);
        else if (isDDown) objs_to_intersect = marked_blocks;
        const intersects = raycasters[2].intersectObjects( objs_to_intersect, false );

        if ( intersects.length > 0 ) {
            const intersect = intersects[ 0 ];
            if (isADown || isDDown){ // overlap existing cubes or marked blocks
                rollOverMesh.position.copy( intersect.object.position );
            } else {  // S down is also the default, with is to show the rollover on top of existing blocks
                rollOverMesh.position.copy( intersect.point ).add( intersect.face.normal );
                rollOverMesh.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
            }
        }
        matchCameras();
        render();
    }
}

function onPointerDown( event ) {
    if (pointers[2]) {
        pointers[2].set( ( (event.clientX - (window.innerWidth/1.9)) / (window.innerWidth/2.1) ) * 2 - 1, - ( event.clientY / (window.innerHeight - 50) ) * 2 + 1 );
        raycasters[2].setFromCamera( pointers[2], cameras[2] );
        
        // Select which blocks the raycaster should intersect with
        let objs_to_intersect = [];
        if (isADown) objs_to_intersect = objects[2];
        else if (isSDown) objs_to_intersect = objects[2].concat(marked_blocks);
        else if (isDDown) objs_to_intersect = marked_blocks;
        const intersects = raycasters[2].intersectObjects( objs_to_intersect, false );

        if ( intersects.length > 0 ) {
            const intersect = intersects[ 0 ];

            // mark cube
            if ( isADown ) {
                if ( intersect.object !== planes[2] ) {
                    // Remove the old block from the scene
                    scenes[2].remove( intersect.object );
                    objects[2].splice( objects[2].indexOf( intersect.object ), 1 );
                    // Add in a marked block in the same spot
                    const voxel = new THREE.Mesh( geo, cubeMaterial_mark );
                    voxel.position.set(intersect.object.position.x,intersect.object.position.y,intersect.object.position.z);
                    scenes[2].add( voxel );
                    marked_blocks.push( voxel );
                    // Record the action to be able to undo
                    actions_taken.push( [intersect.object, voxel, "mark_cube"] );
                }

            // mark air
            } else if (isSDown) {
                // Create new marked block and place on the surface
                const voxel = new THREE.Mesh( geo, cubeMaterial_mark );
                voxel.position.copy( intersect.point ).add( intersect.face.normal );
                voxel.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
                scenes[2].add( voxel );
                marked_blocks.push( voxel );
                // Record the action to be able to undo
                actions_taken.push( [null, voxel, "mark_air"] );

            // unmark block
            } else if (isDDown) {
                if ( intersect.object !== planes[2] ) {
                    // Remove the marked block from the scene
                    scenes[2].remove( intersect.object );
                    marked_blocks.splice( marked_blocks.indexOf( intersect.object ), 1 );
                    // If it existed originally, replace with the original object
                    let voxel = null;
                    objects[1].forEach(obj => {
                        if (obj.position.equals(intersect.object.position)) {
                            voxel = obj.clone()
                            scenes[2].add( voxel );
                            objects[2].push( voxel );
                        }
                    });
                    // Record the action to be able to undo
                    actions_taken.push( [intersect.object, voxel, "unmark_block"] );
                }
            }

            // Signal to the HIT that annotation was at least attempted
            if ((marked_blocks.length > 0) && (!startedHIT)) {
                startedHIT = true;
                window.parent.postMessage(JSON.stringify({ msg: "block_marked" }), "*");
            }
        
            // Store marked blocks in parent HTML.
            inst_seg_tags[0]["locs"] = [];
            marked_blocks.forEach((block) => {
                let positionArray = block.position.toArray();
                let scaledArray = positionArray.map(function(item) { return (item-25)/50 });
                scaledArray[0] += origin_offset;  // Reset the origin in x and z
                scaledArray[2] += origin_offset;
                inst_seg_tags[0]["locs"].push(scaledArray);
            })
            document.getElementById("inst_seg_tags").value = JSON.stringify(inst_seg_tags);

            render();
        }
    }
}

function onDocumentKeyDown( event ) {
    switch ( event.keyCode ) {
        case 65: isADown = true; break;
        case 83: isSDown = true; break;
        case 68: isDDown = true; break;
        case 71: // g to toggle ground visibility
            for (const idx in scenes) {
                scenes[idx].traverse(function(obj){
                    if (obj.name.slice(-2) === "46" || obj.name.slice(-2) === "_8" || obj.name.slice(-2) === "_9") {
                        if (obj.visible) {
                            obj.visible = false;
                            objects[idx].splice( objects[idx].indexOf( obj ), 1 );
                        } else {
                            obj.visible = true;
                            objects[idx].push(obj);
                        }
                    };
                });
            };
            render();
            break;
        case 90:  // z to undo
            let action = actions_taken.pop();
            if (action){
                switch (action[2]) {
                    case "mark_cube":
                        // Remove the marked block
                        scenes[2].remove( action[1] );
                        marked_blocks.splice( marked_blocks.indexOf( action[1] ), 1 );
                        // Put back the original cube
                        scenes[2].add( action[0] );
                        objects[2].push( action[0] );
                        break;
                    case "mark_air":
                        // Remove the marked block
                        scenes[2].remove( action[1] );
                        marked_blocks.splice( marked_blocks.indexOf( action[1] ), 1 );
                        break;
                    case "unmark_block":
                        // If there's a cube there, remove it
                        if (action[1]){
                            scenes[2].remove( action[1] );
                            objects[2].splice( objects[2].indexOf( action[1] ), 1 );
                        }
                        // Put back the marked block
                        scenes[2].add( action[0] );
                        marked_blocks.push( action[0] );
                        break;
                }
                render();
            }
            break;
    }
}

function onDocumentKeyUp( event ) {
    switch ( event.keyCode ) {
        case 65: isADown = false; break;
        case 83: isSDown = false; break;
        case 68: isDDown = false; break;
    }
}

function render() {
    renderers[1].render( scenes[1], cameras[1] );
    if (renderers[2]) renderers[2].render( scenes[2], cameras[2] );
}