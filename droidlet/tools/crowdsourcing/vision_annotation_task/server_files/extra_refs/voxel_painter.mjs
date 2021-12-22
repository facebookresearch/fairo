import * as THREE from 'https://cdn.skypack.dev/three';
import { OrbitControls } from './OrbitControls.mjs';

let camera1, controls1, scene1, renderer1, plane1;
let camera2, controls2, scene2, renderer2, plane2;

let pointer2, raycaster2; 
let isADown = false;
let isSDown = false;
let isDDown = false;

let userRaycaster1, userRaycaster2;

let rollOverMesh2, rollOverMaterial2;
let cubeMaterial_mark;
const geo = new THREE.BoxGeometry( 50, 50, 50 );
const cubeMaterial = new THREE.MeshLambertMaterial( { color: 0xfeb74c, map: new THREE.TextureLoader().load( 'square-outline-textured.png' ) } );
const userMaterial = new THREE.MeshBasicMaterial( { color: 0xffff00, opacity: 1.0} );
const agentMaterial = new THREE.MeshBasicMaterial( { color: 0x0000ff, opacity: 1.0} );
const groundMaterial = new THREE.MeshBasicMaterial( { color: 0xffffff, opacity: 1.0} );

let objects1 = [];
let objects2  = [];
let marked_blocks = [];

let actions_taken = []; // [original_block, new_block, action_type]
var startedHIT = false;

// User position comes from properties.avatarInfo.properties.pos
let user = [[5,5,5,'u1'], [5,6,5,'u2']];
// Agent position comes from ???
let agent = [[5,5,-5,'a1'], [5,6,-5,'a2']];
// Look direction comes in the file properties.avatarInfo.properties.look as [pitch, yaw] floats
// referenced s.t. pos pitch is down, 0 yaw = pos-z, and pos yaw is CCW
// Three.js has pos pitch up, 0 yaw = pos-x, and pos yaw is CW
const raw_look_vals = [Math.PI/4, 1.55*Math.PI]  // [pitch, yaw]
const look_angles = [(-1)*raw_look_vals[0], ((-1)*raw_look_vals[1]) + Math.PI/2]
// Convert pitch and yaw radian values to a Vector3 look direction
let user_look_dir = new THREE.Vector3( 
    Math.cos(look_angles[1]) * Math.cos(look_angles[0]),
    Math.sin(look_angles[0]),
    Math.sin(look_angles[1] * Math.cos(look_angles[0])));
user_look_dir.normalize();
const head_position = new THREE.Vector3((user[1][0]*50)+25, (user[1][1]*50)+25, (user[1][2]*50)+25);

let starting_cube = [];
let block_id = 0;
for (let x=-2; x<2; x++){
    for (let y=0; y<4; y++){
        for (let z=-2; z<2; z++){
            starting_cube.push([x,y,z,block_id++])
        }
    }
}

// TODO download data from S3 based on data.csv?
// In the meantime, here's some dummy data

fetch('./scene1.json')
    .then(response => {
        return response.json();
    })
    .then(jsondata => {
        return jsondata[0];
    })
    .then(scene => {
        init1(scene);
        init2(scene);
        render();
        var canvii = document.getElementsByTagName("canvas");
        Array.from(canvii).forEach(canv => canv.style.display = "inline");
        canvii[1].style.float = "right";
    })

function init1(scene) {
    // This is the left scene, uneditable
    camera1 = new THREE.PerspectiveCamera( 50, (window.innerWidth/2) / (window.innerHeight - 50), 1, 10000 );
    camera1.position.set( 400, 640, 1040 );
    camera1.lookAt( 0, 0, 0 );

    scene1 = new THREE.Scene();
    scene1.background = new THREE.Color( 0xf0f0f0 );

    // grid
    //const gridHelper = new THREE.GridHelper( scene.length*50 , scene.length );
    //scene1.add( gridHelper );

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    plane1 = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene1.add( plane1 );
    objects1.push( plane1 );

    // starting cube
    /*
    starting_cube.forEach((block) => {
        const voxel = new THREE.Mesh( geo, cubeMaterial );
        voxel.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene1.add( voxel );
        objects1.push( voxel );
    })
    */

    // load scene
    let origin_offset = Math.floor(scene.length/2);
    for (let i=0; i<scene.length; i++) {
        for (let j=0; j<scene[i].length; j++) {
            for (let k=0; k<scene[i][j].length; k++) {
                if (scene[i][j][k][0]) {
                    let voxel;
                    if (scene[i][j][k][0] === 35) {  // if it's a shape, not ground
                        voxel = new THREE.Mesh( geo, cubeMaterial );
                    } else {  //assume it's ground, base case
                        voxel = new THREE.Mesh( geo, groundMaterial );
                        const edges = new THREE.EdgesGeometry( geo );  // outline the white blocks for visibility
                        const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x000000 } ) );
                        line.position.set(((i - origin_offset)*50)+25, (j*50)+25, ((k - origin_offset)*50)+25);
                        scene1.add( line );
                    }
                    voxel.position.set(((i - origin_offset)*50)+25, (j*50)+25, ((k - origin_offset)*50)+25);
                    scene1.add( voxel );
                    objects1.push( voxel );
                }
            }
        }
    }

    // add axes helper
    //The X axis is red. The Y axis is green. The Z axis is blue.
    scene1.add( new THREE.AxesHelper( 10000 ) );

    // add look direction and init look raycaster - MUST BE AFTER SCENE, BEFORE USER
    scene1.add( new THREE.ArrowHelper( user_look_dir, head_position, 150, 0xff0000, 40, 20 ) );
    userRaycaster1 = new THREE.Raycaster(head_position, user_look_dir);
    const intersects = userRaycaster1.intersectObjects( objects1, false );
    if ( intersects.length > 0 ) {
        console.log("user raycaster hit something");
        const intersect = intersects[ 0 ];
        intersect.object.material.color.set( 0xffff00 );
    }

    // add user
    user.forEach((block) => {
        const user_block = new THREE.Mesh( geo, userMaterial );
        user_block.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene1.add( user_block );
        objects1.push( user_block );
    })

    // add agent
    agent.forEach((block) => {
        const agent_block = new THREE.Mesh( geo, agentMaterial );
        agent_block.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene1.add( agent_block );
        objects1.push( agent_block );
    });

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

    // controls
    controls1 = new OrbitControls( camera1, renderer1.domElement );
    controls1.listenToKeyEvents( window );
    controls1.addEventListener( 'change', render );

    controls1.enableZoom = false;
    controls1.minPolarAngle = (0.5 * Math.PI) / 4;
    controls1.maxPolarAngle = (2.0 * Math.PI) / 4;

    document.addEventListener( 'keydown', onDocumentKeyDown );
    document.addEventListener( 'keyup', onDocumentKeyUp );
    window.addEventListener( 'resize', onWindowResize );
}

function init2(scene) {
    // This is the right scene, editable
    camera2 = new THREE.PerspectiveCamera( 50, (window.innerWidth/2) / (window.innerHeight - 50), 1, 10000 );
    camera2.position.set( 400, 640, 1040 );
    camera2.lookAt( 0, 0, 0 );

    scene2 = new THREE.Scene();
    scene2.background = new THREE.Color( 0xf0f0f0 );

    // roll-over helpers
    rollOverMaterial2 = new THREE.MeshBasicMaterial( { color: 0xff0000, opacity: 0.3, transparent: true } );
    rollOverMesh2 = new THREE.Mesh( geo, rollOverMaterial2 );
    scene2.add( rollOverMesh2 );

    // marked block material
    cubeMaterial_mark = new THREE.MeshBasicMaterial( { color: 0x089000, opacity: 0.6, transparent: true } );

    // grid
    //const gridHelper = new THREE.GridHelper( scene.length*50, scene.length );
    //scene2.add( gridHelper );

    // pointer for clicking
    raycaster2 = new THREE.Raycaster();
    pointer2 = new THREE.Vector2();

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    plane2 = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scene2.add( plane2 );
    objects2.push( plane2 );

    // starting cube
    /*
    starting_cube.forEach((block) => {
        const voxel = new THREE.Mesh( geo, cubeMaterial );
        voxel.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene2.add( voxel );
        objects2.push( voxel );
    })
    */

    // load scene
    let origin_offset = Math.floor(scene.length/2);
    for (let i=0; i<scene.length; i++) {
        for (let j=0; j<scene[i].length; j++) {
            for (let k=0; k<scene[i][j].length; k++) {
                if (scene[i][j][k][0]) {
                    let voxel;
                    if (scene[i][j][k][0] === 35) {  // if it's a shape, not ground
                        voxel = new THREE.Mesh( geo, cubeMaterial );
                    } else {  //assume it's ground, base case
                        voxel = new THREE.Mesh( geo, groundMaterial );
                        const edges = new THREE.EdgesGeometry( geo );  // outline the white blocks for visibility
                        const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x000000 } ) );
                        line.position.set(((i - origin_offset)*50)+25, (j*50)+25, ((k - origin_offset)*50)+25);
                        scene2.add( line );
                    }
                    voxel.position.set(((i - origin_offset)*50)+25, (j*50)+25, ((k - origin_offset)*50)+25);
                    scene2.add( voxel );
                    objects2.push( voxel );
                }
            }
        }
    }

    // add axes helper
    //The X axis is red. The Y axis is green. The Z axis is blue.
    scene2.add( new THREE.AxesHelper( 10000 ) );

    // add user
    user.forEach((block) => {
        const user_block = new THREE.Mesh( geo, userMaterial );
        user_block.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene2.add( user_block );
        objects2.push( user_block );
    })

    // add look direction
    //user_look_dir.subVectors(scene2.position, head_position).normalize();
    const arrowHelper = new THREE.ArrowHelper( user_look_dir, head_position, 150, 0xff0000, 40, 20 );
    scene2.add( arrowHelper );

    // add agent
    agent.forEach((block) => {
        const agent_block = new THREE.Mesh( geo, agentMaterial );
        agent_block.position.set((block[0]*50)+25, (block[1]*50)+25, (block[2]*50)+25);
        scene2.add( agent_block );
        objects2.push( agent_block );
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

    // controls
    controls2 = new OrbitControls( camera2, renderer2.domElement );
    controls2.listenToKeyEvents( window );
    controls2.addEventListener( 'change', render );

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
    camera1.aspect = (window.innerWidth/2) / (window.innerHeight - 50);
    camera1.updateProjectionMatrix();
    renderer1.setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    camera2.aspect = (window.innerWidth/2) / (window.innerHeight - 50);
    camera2.updateProjectionMatrix();
    renderer2.setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    render();
}

function onPointerMove( event ) {
    pointer2.set( ( (event.clientX - (window.innerWidth/2)) / (window.innerWidth/2) ) * 2 - 1, - ( event.clientY / (window.innerHeight - 50) ) * 2 + 1 );
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
    pointer2.set( ( (event.clientX - (window.innerWidth/2)) / (window.innerWidth/2) ) * 2 - 1, - ( event.clientY / (window.innerHeight - 50) ) * 2 + 1 );
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
                const voxel = new THREE.Mesh( geo, cubeMaterial_mark );
                voxel.position.set(intersect.object.position.x,intersect.object.position.y,intersect.object.position.z);
                scene2.add( voxel );
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
                // If it existed originally, replace with a the original object
                let voxel = null;
                objects1.forEach(obj => {
                    if (obj.position.equals(intersect.object.position)) {
                        voxel = obj.clone()
                        scene2.add( voxel );
                        objects2.push( voxel );
                    }
                });
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

    let output_list = [];
    marked_blocks.forEach((block) => {
        let positionArray = block.position.toArray();
        let scaledArray = positionArray.map(function(item) { return (item-25)/50 });
        output_list.push(scaledArray);
    })
    document.getElementById("markedBlocks").value = JSON.stringify(output_list);
}