/* jshint esversion:8, globalstrict:true, node:true */
"use strict";

async function main() {
    const fs = require("fs");
    const process = require("process");
    const path = require("path");

    const pyodide = await require("pyodide").loadPyodide();
    await pyodide.loadPackage(["scipy", "pytest"]);
    // clapack/openblas should be loaded manually before loading TAT
    // but some version of pyodide use clapack but other use openblas
    // load scipy directly since it always depend on clapack or openblas
    const dist_dir = "dist";
    const files = await fs.promises.readdir(dist_dir);
    await pyodide.loadPackage(path.join(dist_dir, files[0]));

    const mount_dir = "/tests";
    pyodide.FS.mkdir(mount_dir);
    pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, {
        root: "tests"
    }, mount_dir);
    const result = await pyodide.runPython(`
import TAT
print(TAT())

import pytest
pytest.main(["${mount_dir}"])
    `);
    process.exit(result);
}
main();
