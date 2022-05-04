// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';

import { ChickenMob } from './VoxelMob.mjs';

let camera, scene, renderer, controls;
let plane;

const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

// Define cube geometry and materials
const geo = new THREE.BoxGeometry( 50, 50, 50 );

init();
addEventListeners();
addTestContent();
render();

var canvii = document.getElementsByTagName("canvas");
Array.from(canvii).forEach((canv) => {
    canv.style.display = "inline";
    canv.style.margin = "auto";
});


async function addTestContent() {
    console.log("starting test content")
    // hard code test here to run in init
    let world = {
        THREE: THREE,
        scene: scene,
    };
    let opts = {
        GLTFLoader: GLTFLoader,
    };
    ChickenMob.build(world, opts).then(
        function (chicken) {
            console.log(chicken);
            window.setInterval(moveObj, 1000, chicken, 50);
        }
    );
}

function moveObj(obj, dist) {
    let dir = Math.floor(3 * Math.random());
    let choices = [-1, 1];
    let move = choices[Math.floor(choices.length * Math.random())] * dist;
    switch (dir) {
        case 0:
            obj.mob.position.x += move;
            break;
        case 1:
            // obj.mob.position.y += move;
            break;
        case 2:
            obj.mob.position.z += move;
            break;
    }
    render();
}

function init() {

    //camera
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 10000 );
    camera.position.set( 500, 800, 1300 );
    camera.lookAt( 0, 0, 0 );

    scene = new THREE.Scene();
    scene.background = new THREE.Color( 0xf0f0f0 );

    // grid
    const gridHelper = new THREE.GridHelper( 1000, 20 );
    scene.add( gridHelper );
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    plane = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene.add( plane );

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