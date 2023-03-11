#!/usr/bin/env wolframscript
(*
 * Analytical solution for gibbs state of hubbard model.
 *)

parseArgv[i_] := StringCases[$ScriptCommandLine[[i]], x:NumberString:>ToExpression[x]][[1]];

L1 = parseArgv[2];
L2 = parseArgv[3];
orbit = 2;
t = 1;
U = parseArgv[4];
mu = parseArgv[5];

Print[L1," * ",L2," square lattice hubbard model with U=", U, " mu=", mu];

createAt[n_, e[i_]] := Module[{this = 2^n},
  If[BitAnd[this, i] == 0,
     e[i + this] (-1)^DigitCount[BitShiftRight[i, n + 1], 2, 1],
     0]]
particleNumber = L1*L2*orbit;
creation = Table[SparseArray[
        Table[{i, createAt[p, e[i]]}, {i, 0, 2^particleNumber - 1}] /.
        {{n_, 0} -> Nothing, {i_, e[j_]} -> {j + 1, i + 1} -> 1, {i_, -e[j_]} -> {j + 1, i + 1} -> -1},
        {2^particleNumber, 2^particleNumber}], {p, 0, particleNumber - 1}];
annihilation = Table[Transpose[i], {i, creation}];
number = Table[creation[[i]] . annihilation[[i]], {i, 1, particleNumber}];
select[op_, i_, j_, o_] := op[[((i - 1)*L2 + (j - 1))*orbit + (o - 1) + 1]];

H = U Sum[
        select[number, l1, l2, 1] . select[number, l1, l2, 2],
        {l1, L1}, {l2, L2}] -
    mu Sum[
         select[number, l1, l2, 1] + select[number, l1, l2, 2],
         {l1, L1}, {l2, L2}] -
    t (Sum[
         select[creation, l1, l2, 1] . select[annihilation, l1, l2 + 1, 1] +
         select[creation, l1, l2, 2] . select[annihilation, l1, l2 + 1, 2] +
         select[creation, l1, l2 + 1, 1] . select[annihilation, l1, l2, 1] +
         select[creation, l1, l2 + 1, 2] . select[annihilation, l1, l2, 2],
         {l1, L1}, {l2, L2 - 1}] +
       Sum[
         select[creation, l1, l2, 1] . select[annihilation, l1 + 1, l2, 1] +
         select[creation, l1, l2, 2] . select[annihilation, l1 + 1, l2, 2] +
         select[creation, l1 + 1, l2, 1] . select[annihilation, l1, l2, 1] +
         select[creation, l1 + 1, l2, 2] . select[annihilation, l1, l2, 2],
         {l1, L1 - 1}, {l2, L2}]);

eigv = Eigenvalues[N[Normal[H]]] // Sort;
Export[ToString[L1]<>"x"<>ToString[L2]<>"-U"<>ToString[U]<>"-mu"<>ToString[mu]<>".dat", eigv];
