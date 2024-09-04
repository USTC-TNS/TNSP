/* jshint esversion:8, globalstrict:true, node:true */
"use strict";

async function main() {
    const process = require("process");

    const pyodide = await require("pyodide").loadPyodide();
    await pyodide.loadPackage("micropip");

    const mount_dir = "/app";
    pyodide.FS.mkdir(mount_dir);
    pyodide.FS.mount(pyodide.FS.filesystems.NODEFS, {
        root: "."
    }, mount_dir);

    const result = await pyodide.runPython(`
async def main():
    import micropip

    await micropip.install("openblas")

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
