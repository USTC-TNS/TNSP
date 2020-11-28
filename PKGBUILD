#!/usr/bin/env bash
_pkgname=TAT
pkgname=$_pkgname-git
pkgver=326.c1ab99b
pkgrel=1
pkgdesc="TAT is A Tensor library"
arch=('x86_64')
url="https://github.com/hzhangxyz/TAT"
license=('GPL3')
depends=('lapack' 'openblas' 'openmpi' 'python')
makedepends=('cmake' 'gcc' 'pybind11')
optdepends=('emscripten' 'python-numpy')
provides=('TAT')
source=(git+https://github.com/hzhangxyz/$_pkgname.git)
md5sums=('SKIP')
pkgver() {
   echo $(git rev-list --count HEAD).$(git rev-parse --short HEAD)
}
_get_package_version() {
   pacman -Q $1 | awk -F'[ .]' '{print $2"."$3}'
}
build() {
   cd $srcdir/$_pkgname
   sed -i s/dev/$pkgver/ FindTAT.cmake
   sed -i s/unknown/$pkgver/ include/TAT/TAT.hpp
   rm -rf build
   mkdir build
   cd build
   cmake .. -DBLA_VENDOR=Generic -DCMAKE_BUILD_TYPE=Release
   make PyTAT
}
package() {
   local pythonver=$(_get_package_version python)
   local cmakever=$(_get_package_version cmake)
   install -dm755 $pkgdir/usr/include/TAT
   install -m644 $srcdir/$_pkgname/include/TAT/*.hpp $pkgdir/usr/include/TAT/
   install -dm755 $pkgdir/usr/share/cmake-$cmakever/Modules
   install -m644 $srcdir/$_pkgname/FindTAT.cmake $pkgdir/usr/share/cmake-$cmakever/Modules/
   install -dm755 $pkgdir/usr/lib/python$pythonver/site-packages
   install -m755 $srcdir/$_pkgname/build/TAT.*.so $pkgdir/usr/lib/python$pythonver/site-packages/
}
