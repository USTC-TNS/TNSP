vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO hzhangxyz/TAT
  HEAD_REF dev
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS -DTAT_BUILD_PYTAT=OFF -DTAT_BUILD_TEST=OFF
)

vcpkg_cmake_install()

file(INSTALL "${SOURCE_PATH}/LICENSE.rst" DESTINATION "${CURRENT_PACKAGES_DIR}/share/tat" RENAME copyright)

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/TAT) # .cmake moved to correct folder
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share") # only license here
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/licenses") # license moved already
