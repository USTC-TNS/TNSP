vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO USTC-TNS/TNSP
  HEAD_REF main
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}/TAT"
)

vcpkg_cmake_install()

file(INSTALL "${SOURCE_PATH}/LICENSE.rst" DESTINATION "${CURRENT_PACKAGES_DIR}/share/tat" RENAME copyright)

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/TAT) # .cmake moved to correct folder
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share") # only license here
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/licenses") # license moved already
