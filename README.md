# [TAT](https://github.com/hzhangxyz/TAT) &middot; [![version](https://img.shields.io/github/release/hzhangxyz/TAT.svg)](https://github.com/hzhangxyz/TAT/releases/latest) [![license](https://img.shields.io/github/license/hzhangxyz/TAT.svg)](https://github.com/hzhangxyz/TAT/blob/TAT/LICENSE) ![platform](https://img.shields.io/badge/platform-linux-brightgreen.svg) ![language](https://img.shields.io/badge/language-c++-yellow.svg) [![build](https://travis-ci.com/hzhangxyz/TAT.svg?branch=TAT)](https://travis-ci.com/hzhangxyz/TAT)

TAT is A Tensor library

## current function
- a TAT lib
- a MPS demo

## dependence

| Repo                                                                                                           | License                                                                                                                                     |
|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| [![dependence](https://img.shields.io/badge/Taywee-args-blue.svg)](https://github.com/Taywee/args)             | [![license](https://img.shields.io/github/license/Taywee/args.svg)](https://github.com/Taywee/args/blob/master/LICENSE)                     |
| [![dependence](https://img.shields.io/badge/springer13-hptt-blue.svg)](https://github.com/springer13/hptt)     | [![license](https://img.shields.io/github/license/springer13/hptt.svg)](https://github.com/springer13/hptt/blob/master/LICENSE.txt)         |
| [![dependence](https://img.shields.io/badge/agauniyal-rang-blue.svg)](https://github.com/agauniyal/rang)       | [![license](https://img.shields.io/github/license/agauniyal/rang.svg)](https://github.com/agauniyal/rang/blob/master/LICENSE)               |
| [![dependence](https://img.shields.io/badge/jemalloc-jemalloc-blue.svg)](https://github.com/jemalloc/jemalloc) | [![license](https://img.shields.io/github/license/jemalloc/jemalloc.svg)](https://github.com/jemalloc/jemalloc/blob/dev/COPYING)            |
| [![dependence](https://img.shields.io/badge/intel-mkl-blue.svg)](https://software.intel.com/en-us/mkl)         | [![license](https://img.shields.io/badge/license-ISSL-red.svg)](https://software.intel.com/en-us/license/intel-simplified-software-license) |

---

## TODO LIST
- dgegqr magma                                                    [等待cuda]
- cuda，sw                                                        [等待cpu完善]
- Truncated SVD                                                   [无重要性]
- 使用, peps, kitaev, hubbard                                   [等待lazy与lattice]
- 分块tensor, fermi tensor
- 转置不使用hptt，自己写也许更好，但是比较难                                     [比较重要]
- contract中, 可能使用?dot, ?gemv,以及转置策略问题, 类似的lq问题     [比较重要]
- multiple与contract的融合，pess中经常使用
- lazy tensor 自动update                                    [WIP]
- doxygen
