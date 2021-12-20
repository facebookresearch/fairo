import * as THREE from 'https://cdn.skypack.dev/three';
import { OrbitControls } from './OrbitControls.mjs';

let camera1, controls1, scene1, renderer1, plane1;
let camera2, controls2, scene2, renderer2, plane2;

let pointer2, raycaster2; 
let isADown = false;
let isSDown = false;
let isDDown = false;

let rollOverMesh2, rollOverMaterial2;
let cubeGeo1, cubeMaterial1;
let cubeGeo2, cubeMaterial2;
let cubeGeo_mark, cubeMaterial_mark;

let objects1 = [];
let objects2  = [];
let marked_blocks = [];

let starting_cube = [];
let block_id = 0;
for (let x=-2; x<2; x++){
    for (let y=0; y<4; y++){
        for (let z=-2; z<2; z++){
            starting_cube.push([x,y,z,block_id++])
        }
    }
}

let actions_taken = []; // [original_object, new_object, action_type]
var startedHIT = false;

// TODO download data from S3 based on data.csv?

init1();
init2();
render();

var canvii = document.getElementsByTagName("canvas");
Array.from(canvii).forEach(canv => canv.style.display = "inline");
canvii[1].style.float = "right";

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

    // place
    plane1 = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene1.add( plane1 );
    objects1.push( plane1 );

    // starting cube
    starting_cube.forEach((block) => {
        const voxel = new THREE.Mesh( cubeGeo1, cubeMaterial1 );
        voxel.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene1.add( voxel );
        objects1.push( voxel );
    })

    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scene1.add( ambientLight );
    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scene1.add( directionalLight );

    renderer1 = new THREE.WebGLRenderer( { antialias: true } );
    renderer1.setPixelRatio( window.devicePixelRatio );
    renderer1.setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    let cont = document.getElementById("voxel_painter");
    cont.appendChild( renderer1.domElement );

    controls1 = new OrbitControls( camera1, renderer1.domElement );
    controls1.listenToKeyEvents( window ); // optional
    controls1.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

    controls1.enableZoom = false;
    controls1.minPolarAngle = (0.5 * Math.PI) / 4;
    controls1.maxPolarAngle = (2.0 * Math.PI) / 4;

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
    rollOverMaterial2 = new THREE.MeshBasicMaterial( { color: 0xff0000, opacity: 0.3, transparent: true } );
    rollOverMesh2 = new THREE.Mesh( rollOverGeo, rollOverMaterial2 );
    scene2.add( rollOverMesh2 );

    // marked block params
    cubeGeo_mark = new THREE.BoxGeometry( 50, 50, 50 );
    cubeMaterial_mark = new THREE.MeshBasicMaterial( { color: 0x089000, opacity: 0.6, transparent: true } );

    // cubes
    cubeGeo2 = new THREE.BoxGeometry( 50, 50, 50 );
    cubeMaterial2 = new THREE.MeshLambertMaterial( { color: 0xfeb74c, map: new THREE.TextureLoader().load( 'square-outline-textured.png' ) } );

    // grid
    const gridHelper = new THREE.GridHelper( 1000, 20 );
    scene2.add( gridHelper );
    raycaster2 = new THREE.Raycaster();
    pointer2 = new THREE.Vector2();

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    plane2 = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene2.add( plane2 );
    objects2.push( plane2 );

    // starting cube
    starting_cube.forEach((block) => {
        const voxel = new THREE.Mesh( cubeGeo2, cubeMaterial2 );
        voxel.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene2.add( voxel );
        objects2.push( voxel );
    })

    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scene2.add( ambientLight );
    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scene2.add( directionalLight );

    renderer2 = new THREE.WebGLRenderer( { antialias: true } );
    renderer2.setPixelRatio( window.devicePixelRatio );
    renderer2.setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    let cont = document.getElementById("voxel_painter");
    cont.appendChild( renderer2.domElement );

    controls2 = new OrbitControls( camera2, renderer2.domElement );
    controls2.listenToKeyEvents( window ); // optional
    controls2.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)

    controls2.enableZoom = false;
    controls2.minPolarAngle = (0.5 * Math.PI) / 4;
    controls2.maxPolarAngle = (2.0 * Math.PI) / 4;

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
    pointer2.set( ( (event.clientX - (window.innerWidth/2)) / (window.innerWidth/2) ) * 2 - 1, - ( event.clientY / window.innerHeight ) * 2 + 1 );
    raycaster2.setFromCamera( pointer2, camera2 );
    // Select which blocks the raycaster should intersect with
    let objs_to_intersect = objects2;
    if (isSDown) objs_to_intersect = objects2.concat(marked_blocks);
    else if (isDDown) objs_to_intersect = marked_blocks;
    const intersects = raycaster2.intersectObjects( objs_to_intersect, false );

    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];
        if (isADown || isDDown){ // overlap existing cubes or marked blocks
            rollOverMesh2.position.copy( intersect.object.position );
        } else {  // S down is also the default, with is to show the rollover on top of existing blocks
            rollOverMesh2.position.copy( intersect.point ).add( intersect.face.normal );
            rollOverMesh2.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
        }
        
    }
    render();
}

function onPointerDown( event ) {
    pointer2.set( ( (event.clientX - (window.innerWidth/2)) / (window.innerWidth/2) ) * 2 - 1, - ( event.clientY / window.innerHeight ) * 2 + 1 );
    raycaster2.setFromCamera( pointer2, camera2 );
    
    // Select which blocks the raycaster should intersect with
    let objs_to_intersect = [];
    if (isADown) objs_to_intersect = objects2;
    else if (isSDown) objs_to_intersect = objects2.concat(marked_blocks);
    else if (isDDown) objs_to_intersect = marked_blocks;
    const intersects = raycaster2.intersectObjects( objs_to_intersect, false );

    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];

        // mark cube
        if ( isADown ) {
            if ( intersect.object !== plane2 ) {
                // Remove the old block from the scene
                scene2.remove( intersect.object );
                objects2.splice( objects2.indexOf( intersect.object ), 1 );
                // Add in a marked block in the same spot
                const voxel = new THREE.Mesh( cubeGeo_mark, cubeMaterial_mark );
                voxel.position.set(intersect.object.position.x,intersect.object.position.y,intersect.object.position.z);
                scene2.add( voxel );
                marked_blocks.push( voxel );
                // Record the action to be able to undo
                actions_taken.push( [intersect.object, voxel, "mark_cube"] );
            }

        // mark air
        } else if (isSDown) {
            // Create new marked block and place on the surface
            const voxel = new THREE.Mesh( cubeGeo_mark, cubeMaterial_mark );
            voxel.position.copy( intersect.point ).add( intersect.face.normal );
            voxel.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
            scene2.add( voxel );
            marked_blocks.push( voxel );
            // Record the action to be able to undo
            actions_taken.push( [null, voxel, "mark_air"] );

        // unmark block
        } else if (isDDown) {
            if ( intersect.object !== plane2 ) {
                // Remove the marked block from the scene
                scene2.remove( intersect.object );
                marked_blocks.splice( marked_blocks.indexOf( intersect.object ), 1 );
                // If it existed originally, replace with a cube
                let replace_with_block = false;
                objects1.forEach(obj => {
                    if (obj.position.equals(intersect.object.position)) {
                        replace_with_block = true;
                    }
                });
                let voxel = null;
                if (replace_with_block) {
                    voxel = new THREE.Mesh( cubeGeo2, cubeMaterial2 );
                    voxel.position.set(intersect.object.position.x,intersect.object.position.y,intersect.object.position.z);
                    scene2.add( voxel );
                    objects2.push( voxel );
                }
                // Record the action to be able to undo
                actions_taken.push( [intersect.object, voxel, "unmark_block"] );
            }
        }
        render();
    }
}

function onDocumentKeyDown( event ) {
    switch ( event.keyCode ) {
        case 65: isADown = true; break;
        case 83: isSDown = true; break;
        case 68: isDDown = true; break;
        case 90:  // ctrl-z to undo
            let action = actions_taken.pop();
            if (action){
                switch (action[2]) {
                    case "mark_cube":
                        // Remove the marked block
                        scene2.remove( action[1] );
                        marked_blocks.splice( marked_blocks.indexOf( action[1] ), 1 );
                        // Put back the original cube
                        scene2.add( action[0] );
                        objects2.push( action[0] );
                        break;
                    case "mark_air":
                        // Remove the marked block
                        scene2.remove( action[1] );
                        marked_blocks.splice( marked_blocks.indexOf( action[1] ), 1 );
                        break;
                    case "unmark_block":
                        // If there's a cube there, remove it
                        if (action[1]){
                            scene2.remove( action[1] );
                            objects2.splice( objects2.indexOf( action[1] ), 1 );
                        }
                        // Put back the marked block
                        scene2.add( action[0] );
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
    renderer1.render( scene1, camera1 );
    renderer2.render( scene2, camera2 );

    if ((marked_blocks.length > 0) && (!startedHIT)) {
        startedHIT = true;
        window.parent.postMessage(JSON.stringify({ msg: "block_marked" }), "*");
    }
}