#!/usr/bin/env bash
pkgname=TAT
pkgver=0.1.0
suffix=-rc.9
TATver=$pkgver$suffix
pkgrel=1
pkgdesc="TAT is A Tensor library"
arch=('x86_64')
url="https://github.com/hzhangxyz/TAT"
license=('GPL3')
depends=('lapack' 'blas' 'openmpi' 'python')
makedepends=('cmake' 'gcc' 'pybind11')
optdepends=('emscripten' 'python-numpy')
pythonver=3.8
cmakever=3.19
source=(https://github.com/hzhangxyz/TAT/archive/v${TATver}.tar.gz)
sha256sums=('SKIP')
build() {
   cd $srcdir/$pkgname-$TATver
   sed -i s/dev/$TATver/ FindTAT.cmake
   mkdir build
   cd build
   cmake .. -DBLA_VENDOR=Generic -DCMAKE_BUILD_TYPE=Release
   make PyTAT
}
package() {
   install -dm755 $pkgdir/usr/include/TAT
   install -m644 $srcdir/$pkgname-$TATver/include/TAT/*.hpp $pkgdir/usr/include/TAT/
   install -dm755 $pkgdir/usr/share/cmake-$cmakever/Modules
   install -m644 $srcdir/$pkgname-$TATver/FindTAT.cmake $pkgdir/usr/share/cmake-$cmakever/Modules/
   install -dm755 $pkgdir/usr/lib/python$pythonver/site-packages
   install -m755 $srcdir/$pkgname-$TATver/build/TAT.*.so $pkgdir/usr/lib/python$pythonver/site-packages/
}
