#!/usr/bin/env wolframscript
(*
 * Analytical solution for free fermion.
 *)

L1s = StringCases[$ScriptCommandLine[[2]], x:NumberString:>ToExpression[x]];
L1 = L1s[[1]];
L2s = StringCases[$ScriptCommandLine[[3]], x:NumberString:>ToExpression[x]];
L2 = L2s[[1]];
Ts = StringCases[$ScriptCommandLine[[4]], x:NumberString:>ToExpression[x]];
T = Ts[[1]];
Print[L1," * ",L2," square lattice free fermion with particle number ", T];

H = ArrayReshape[
        Table[
                If[Total[Abs[{a,b}-{c,d}]]==1,1.,0.],
                {a, L1},
                {b, L2},
                {c, L1},
                {d, L2}
        ],
        {L1*L2,L1*L2}
    ];
Energy = Total[Sort[Eigenvalues[H]][[;;T]]];
Print[Energy/(L1*L2)];
