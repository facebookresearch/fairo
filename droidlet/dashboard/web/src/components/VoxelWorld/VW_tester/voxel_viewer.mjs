// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import * as THREE from './three.module.mjs';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';
import { BLOCK_MAP } from './blockMap.mjs';

let camera, scene, renderer, controls;
let plane;

const minCameraPitch = (0.5 * Math.PI) / 4;
const maxCameraPitch = (2.0 * Math.PI) / 4;

// Define cube geometry and materials
const geo = new THREE.BoxGeometry( 50, 50, 50 );

init();
addEventListeners();
render();
var canvii = document.getElementsByTagName("canvas");
Array.from(canvii).forEach((canv) => {
    canv.style.display = "inline";
    canv.style.margin = "auto";
});


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