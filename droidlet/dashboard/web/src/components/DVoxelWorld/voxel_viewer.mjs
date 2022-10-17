// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';

import { VoxelMob } from './VoxelMob.mjs';
import { VoxelItem } from './VoxelItem.mjs';
import { VoxelPlayer} from './VoxelPlayer.mjs'
import { VW_ITEM_MAP, VW_MOB_MAP, VW_AVATAR_MAP } from './model_luts.mjs'

let camera, scene, renderer, controls, plane, cursorX, cursorY;
let players = [];

const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

// Define cube geometry and materials
const geo = new THREE.BoxGeometry( 50, 50, 50 );

init();
addEventListeners();
addTestContent();
render();
console.log(scene);

var canvii = document.getElementsByTagName("canvas");
Array.from(canvii).forEach((canv) => {
    canv.style.display = "inline";
    canv.style.margin = "auto";
});


function addTestContent() {
    // hard code test here to run in init
    let world = {
        THREE: THREE,
        scene: scene,
        render: render,
        camera: camera,
    };

    for (const key in VW_MOB_MAP) {
        if (typeof(key) === "string" && VW_MOB_MAP[key] !== null) {
            const opts = {
                GLTFLoader: GLTFLoader,
                name: key,
            };
            VoxelMob.build(world, opts).then(
                function (mob) {
                    window.setInterval(walkabout, 1000, mob, 50);
                }
            );
        }
    };

    for (const key in VW_ITEM_MAP) {
        if (typeof(key) === "string" && VW_ITEM_MAP[key] !== null) {
            const opts = {
                GLTFLoader: GLTFLoader,
                name: key,
            };
            VoxelItem.build(world, opts).then(
                function (item) {
                    window.setInterval(walkabout, 1000, item, 50);
                }
            );
        }
    };

    for (const key in VW_AVATAR_MAP) {
        if (typeof(key) === "string" && VW_AVATAR_MAP[key] !== null) {
            const opts = {
                GLTFLoader: GLTFLoader,
                name: key,
            };
            VoxelPlayer.build(world, opts).then(
                function (player) {
                    window.setInterval(walkabout, 1000, player, 50);
                    players.push(player);
                    if (player.avatarType === "player") {
                        cameraTest(player);
                    }
                }
            );
        }
    };

}

function cameraTest(player) {
    controls.enabled = false;
    window.addEventListener("keydown", function (e) {
        handleKeypress(e, player);
    });
    player.possess();
};

function handleKeypress(e, player) {
    switch (e.key) {
        case "ArrowLeft":
            player.rotate(0.1);
            break;
        case "ArrowRight":
            player.rotate(-0.1);
            break;
        case "t":
            player.toggle();
    }
}

function walkabout(obj, dist) {
    let dir = Math.floor(3 * Math.random());
    let choices = [-1, 1];
    let move = choices[Math.floor(choices.length * Math.random())] * dist;
    switch (dir) {
        case 0:
            if (obj.mesh.position.x < 500 && obj.mesh.position.x > -500){
                obj.move(move, 0, 0);
                if (obj instanceof(VoxelItem)) {
                    obj.pick();
                }
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
                if (obj instanceof(VoxelItem)) {
                    obj.drop();
                }
            } else {
                obj.moveTo(0,0,0);
            }
            break;
    }
    render();
}

function init() {

    // camera
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 10000 );
    camera.position.set( 500, 800, 1300 );
    camera.lookAt( 0, 0, 0 );

    // scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xf0f0f0 );

    // grid
    const gridHelper = new THREE.GridHelper( 1000, 20 );
    scene.add( gridHelper );
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    plane = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene.add( plane );

    //Axis helper - The positive X axis is red, Y is green, Z  is blue.
    scene.add( new THREE.AxesHelper( 10000 ) );

    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scene.add( ambientLight );
    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scene.add( directionalLight );

    // renderer
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


function addEventListeners() {
    window.addEventListener( 'resize', onWindowResize );
    document.addEventListener( 'pointermove', onPointerMove );
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
    render();
}

function onPointerMove( event ) {
    let Xdiff = ( cursorX - event.clientX ) / 150;
    let Ydiff = ( cursorY - event.clientY ) / 150;
    
    players.forEach(player => {
        if (player.possessed) {
            player.cameraPitch(Ydiff);
            player.rotate(Xdiff);
        }
    });
    cursorX = event.clientX;
    cursorY = event.clientY;
}

function render() {
    renderer.render( scene, camera );
}