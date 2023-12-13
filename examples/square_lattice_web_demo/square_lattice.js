"use strict";

import * as THREE from "three";
import {
    OrbitControls
} from "three/addons/controls/OrbitControls.js";
import loadWASM from "./kernel.js";

var kernel = await loadWASM();

var renderer = new THREE.WebGLRenderer();
var width = window.innerWidth;
var height = window.innerHeight;
renderer.setSize(width, height);
document.body.appendChild(renderer.domElement);

var scene = new THREE.Scene();

var axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

var geometry = new THREE.SphereGeometry(0.05, 10, 10);
var material = new THREE.MeshNormalMaterial();

// var camera = new THREE.OrthographicCamera(-width/2, +width/2, -height/2, +height/2, 0.1, 10);
var camera = new THREE.PerspectiveCamera(20, width / height, 5, 1000);
camera.position.set(10, 10, 5);
camera.up.set(0, 0, 1);
camera.lookAt(0, 0, 0);

var controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

var do_render = () => {
    controls.update();
    renderer.render(scene, camera);
};
setInterval(do_render, 50);

var z = new THREE.Vector3(0, 0, 1);

var L1 = 0;
var L2 = 0;

document.getElementById("create_lattice").onclick = () => {
    var i, j;
    for (i = 0; i < L1; i++) {
        for (j = 0; j < L2; j++) {
            scene.remove(scene.getObjectByName("S" + i + "." + j));
            scene.remove(scene.getObjectByName("A" + i + "." + j));
        }
    }
    L1 = parseInt(document.getElementById("L1").value);
    L2 = parseInt(document.getElementById("L2").value);
    if (L1 * L2 >= 18) {
        L1 = 0;
        L2 = 0;
        alert("too big system size");
        return
    }
    kernel._create_lattice(L1, L2);
    var offset_L1 = (L1 - 1) / 2;
    var offset_L2 = (L2 - 1) / 2;
    for (i = 0; i < L1; i++) {
        for (j = 0; j < L2; j++) {
            var sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(i - offset_L1, j - offset_L2, 0);
            sphere.name = "S" + i + "." + j;
            scene.add(sphere);

            var arrow = new THREE.ArrowHelper(z, new THREE.Vector3(i - offset_L1, j - offset_L2, 0), 0.5);
            arrow.name = "A" + i + "." + j;
            scene.add(arrow);
        }
    }
}
document.getElementById("update_lattice").onclick = () => {
    if (L1 == 0 || L2 == 0) {
        return;
    }
    var start = performance.now();
    kernel._update_lattice(parseInt(document.getElementById("step").value));
    var energy = kernel._get_energy();
    document.getElementById("energy").value = energy;
    var den = kernel._get_den() / 2;
    for (var i = 0; i < L1; i++) {
        for (var j = 0; j < L2; j++) {
            var sx = kernel._get_spin(i, j, 0);
            var sy = kernel._get_spin(i, j, 1);
            var sz = kernel._get_spin(i, j, 2);
            var dir = new THREE.Vector3(sx, sy, sz);
            var len = dir.length();
            scene.getObjectByName("A" + i + "." + j).setDirection(dir.normalize());
            scene.getObjectByName("A" + i + "." + j).setLength(len / den);
        }
    }
    var end = performance.now();
    console.log((end - start) / 1000);
};
