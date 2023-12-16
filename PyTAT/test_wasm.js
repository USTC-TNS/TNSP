/* jshint esversion:8, globalstrict:true, node:true */
"use strict";

async function main() {
    const process = require("process");

    const pyodide = await require("pyodide").loadPyodide();
    await pyodide.loadPackage("micropip");
    await pyodide.loadPackage("openblas");
    // openblas should be loaded manually before loading TAT
    // see https://github.com/ryanking13/auditwheel-emscripten/issues/24
    // when this been fixed, removing load openblas manually,
    // and uncomment pyodide auditwheel in github action.

    const mount_dir = "/app";
    pyodide.FS.mkdir(mount_dir);
    pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, {
        root: "."
    }, mount_dir);

    const result = await pyodide.runPython(`
async def main():
    import micropip

    import os
    files = os.listdir("/app/dist")
    await micropip.install(f"emfs:/app/dist/{files[0]}")

    import TAT
    print(TAT())

    await micropip.install("pytest")
    import pytest
    return pytest.main(["/app/tests"])


main()
`);
    process.exit(result);
}
main();
