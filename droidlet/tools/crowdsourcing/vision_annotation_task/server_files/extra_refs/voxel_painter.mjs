import * as THREE from 'https://cdn.skypack.dev/three';
import { OrbitControls } from './OrbitControls.mjs';

let camera1, controls1, scene1, renderer1;
let plane1;
let camera2, controls2, scene2, renderer2;
let plane2;

let pointer2, raycaster2; 
let isShiftDown = false;

let rollOverMesh2, rollOverMaterial2;
let cubeGeo1, cubeMaterial1;
let cubeGeo2, cubeMaterial2;

const objects = [];

init1();
init2();
render();

var canvii = document.getElementsByTagName("canvas");
Array.from(canvii).forEach(canv => canv.style.display = "inline");

function init1() {
    // This is the left scene, uneditable
    camera1 = new THREE.PerspectiveCamera( 45, (window.innerWidth/2) / window.innerHeight, 1, 10000 );
    camera1.position.set( 500, 800, 1300 );
    camera1.lookAt( 0, 0, 0 );

    scene1 = new THREE.Scene();
    scene1.background = new THREE.Color( 0xf0f0f0 );

    // cubes
    cubeGeo1 = new THREE.BoxGeometry( 50, 50, 50 );
    cubeMaterial1 = new THREE.MeshLambertMaterial( { color: 0xfeb74c, map: new THREE.TextureLoader().load( 'square-outline-textured.png' ) } );

    // grid
    const gridHelper = new THREE.GridHelper( 1000, 20 );
    scene1.add( gridHelper );

    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );

    plane1 = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene1.add( plane1 );

    objects.push( plane1 );

    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scene1.add( ambientLight );

    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scene1.add( directionalLight );

    renderer1 = new THREE.WebGLRenderer( { antialias: true } );
    renderer1.setPixelRatio( window.devicePixelRatio );
    renderer1.setSize( (window.innerWidth/2.1), window.innerHeight );
    let cont = document.getElementById("voxel_painter");
    cont.appendChild( renderer1.domElement );

    controls1 = new OrbitControls( camera1, renderer1.domElement );
    controls1.listenToKeyEvents( window ); // optional
    controls1.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

    controls1.enableZoom = false;
    controls1.minPolarAngle = (1.2 * Math.PI) / 4;
    controls1.maxPolarAngle = (1.2 * Math.PI) / 4;

    document.addEventListener( 'keydown', onDocumentKeyDown );
    document.addEventListener( 'keyup', onDocumentKeyUp );
    window.addEventListener( 'resize', onWindowResize );
}

function init2() {
    // This is the right scene, uneditable
    camera2 = new THREE.PerspectiveCamera( 45, (window.innerWidth/2) / window.innerHeight, 1, 10000 );
    camera2.position.set( 500, 800, 1300 );
    camera2.lookAt( 0, 0, 0 );

    scene2 = new THREE.Scene();
    scene2.background = new THREE.Color( 0xf0f0f0 );

    // roll-over helpers
    const rollOverGeo = new THREE.BoxGeometry( 50, 50, 50 );
    rollOverMaterial2 = new THREE.MeshBasicMaterial( { color: 0xff0000, opacity: 0.5, transparent: true } );
    rollOverMesh2 = new THREE.Mesh( rollOverGeo, rollOverMaterial2 );
    scene2.add( rollOverMesh2 );

    // cubes
    cubeGeo2 = new THREE.BoxGeometry( 50, 50, 50 );
    cubeMaterial2 = new THREE.MeshLambertMaterial( { color: 0xfeb74c, map: new THREE.TextureLoader().load( 'square-outline-textured.png' ) } );

    // grid
    const gridHelper = new THREE.GridHelper( 1000, 20 );
    scene2.add( gridHelper );
    raycaster2 = new THREE.Raycaster();
    pointer2 = new THREE.Vector2();

    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );

    plane2 = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene2.add( plane2 );

    objects.push( plane2 );

    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scene2.add( ambientLight );

    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scene2.add( directionalLight );

    renderer2 = new THREE.WebGLRenderer( { antialias: true } );
    renderer2.setPixelRatio( window.devicePixelRatio );
    renderer2.setSize( (window.innerWidth/2.1), window.innerHeight );
    let cont = document.getElementById("voxel_painter");
    cont.appendChild( renderer2.domElement );

    controls2 = new OrbitControls( camera2, renderer2.domElement );
    controls2.listenToKeyEvents( window ); // optional
    controls2.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

    controls2.enableZoom = false;
    controls2.minPolarAngle = (1.2 * Math.PI) / 4;
    controls2.maxPolarAngle = (1.2 * Math.PI) / 4;

    document.addEventListener( 'pointermove', onPointerMove );
    document.addEventListener( 'pointerdown', onPointerDown );
    document.addEventListener( 'keydown', onDocumentKeyDown );
    document.addEventListener( 'keyup', onDocumentKeyUp );
    window.addEventListener( 'resize', onWindowResize );
}

function onWindowResize() {
    camera1.aspect = (window.innerWidth/2) / window.innerHeight;
    camera1.updateProjectionMatrix();
    renderer1.setSize( (window.innerWidth/2.1), window.innerHeight );
    camera2.aspect = (window.innerWidth/2) / window.innerHeight;
    camera2.updateProjectionMatrix();
    renderer2.setSize( (window.innerWidth/2.1), window.innerHeight );
    render();
}

function onPointerMove( event ) {
    pointer2.set( ( (event.clientX - (window.innerWidth/2.1)) / (window.innerWidth/2.1) ) * 2 - 1, - ( event.clientY / window.innerHeight ) * 2 + 1 );
    raycaster2.setFromCamera( pointer2, camera2 );
    const intersects = raycaster2.intersectObjects( objects, false );

    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];
        rollOverMesh2.position.copy( intersect.point ).add( intersect.face.normal );
        rollOverMesh2.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
    }
    render();
}

function onPointerDown( event ) {
    pointer2.set( ( (event.clientX - (window.innerWidth/2.1)) / (window.innerWidth/2.1) ) * 2 - 1, - ( event.clientY / window.innerHeight ) * 2 + 1 );
    raycaster2.setFromCamera( pointer2, camera2 );
    const intersects = raycaster2.intersectObjects( objects, false );

    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];

        // delete cube
        if ( isShiftDown ) {
        if ( intersect.object !== plane ) {
            scene2.remove( intersect.object );
            objects.splice( objects.indexOf( intersect.object ), 1 );
        }

        // create cube
        } else {
        const voxel = new THREE.Mesh( cubeGeo2, cubeMaterial2 );
        voxel.position.copy( intersect.point ).add( intersect.face.normal );
        voxel.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
        scene2.add( voxel );
        objects.push( voxel );
        }
        render();
    }
}

function onDocumentKeyDown( event ) {
    switch ( event.keyCode ) {
        case 16: isShiftDown = true; break;
    }
}

function onDocumentKeyUp( event ) {
    switch ( event.keyCode ) {
        case 16: isShiftDown = false; break;
    }
}

function render() {
    renderer1.render( scene1, camera1 );
    renderer2.render( scene2, camera2 );
}