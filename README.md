# [TAT](https://github.com/hzhangxyz/TAT) &middot; [![version](https://img.shields.io/github/release/hzhangxyz/TAT.svg)](https://github.com/hzhangxyz/TAT/releases/latest) [![license](https://img.shields.io/github/license/hzhangxyz/TAT.svg)](https://github.com/hzhangxyz/TAT/blob/TAT/LICENSE) [![build](https://travis-ci.com/hzhangxyz/TAT.svg?branch=TAT)](https://travis-ci.com/hzhangxyz/TAT)

TAT is A Tensor library

## Current Example
- Matrix Product States Simple Update
- Projected Entangled Pair States Simple Update

## Dependence

- [![dependence](https://img.shields.io/badge/Taywee-args-blue.svg)](https://github.com/Taywee/args)
- [![dependence](https://img.shields.io/badge/springer13-hptt-blue.svg)](https://github.com/springer13/hptt)
- [![dependence](https://img.shields.io/badge/agauniyal-rang-blue.svg)](https://github.com/agauniyal/rang)

---

## TODO LIST
- dgegqr magma                                                    [等待cuda]
- cuda，sw                                                        [等待cpu完善]
- Truncated SVD                                                   [无重要性]
- 使用, peps, kitaev, hubbard                                   [等待lazy与lattice]
- 分块tensor, fermi tensor
- 转置不使用hptt，自己写也许更好，但是比较难                                     [比较重要]
- contract中, 可能使用?dot, ?gemv,以及转置策略问题, 类似的lq问题, 顺便还有rq的优化    [比较重要]
- multiple与contract的融合，pess中经常使用
- 不同类型之间的直接scalar，contract等                              [scalar done]
- doxygen
- 缺腿后的稳定性                                                         [基本可以用]
- use it
- tensor和node解耦合, 实现分块tensor的时候再考虑, 不然容易解耦得不好
- lazy framework的内存issue, 深搜变宽搜, 然后按照level依次运算
