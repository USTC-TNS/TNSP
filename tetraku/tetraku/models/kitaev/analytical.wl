#!/usr/bin/env wolframscript
(*
 * Analytical solution for kitaev model.
 *)

L1s = StringCases[$ScriptCommandLine[[2]], x:NumberString:>ToExpression[x]];
L1 = L1s[[1]];
L2s = StringCases[$ScriptCommandLine[[3]], x:NumberString:>ToExpression[x]];
L2 = L2s[[1]];
Jxs = StringCases[$ScriptCommandLine[[4]], x:NumberString:>ToExpression[x]];
Jx = N[Jxs[[1]]];
Jys = StringCases[$ScriptCommandLine[[5]], x:NumberString:>ToExpression[x]];
Jy = N[Jys[[1]]];
Jzs = StringCases[$ScriptCommandLine[[6]], x:NumberString:>ToExpression[x]];
Jz = N[Jzs[[1]]];
Print[L1," * ",L2," * 2 - 2 kitaev model with J = ", Jx, ", ", Jy, ", ", Jz];

siteNumber = L1*L2*2-2;
getIndex[l1_,l2_,orbit_]:=((l1-1)*L2+(l2-1))*2+(orbit-1);
H = ConstantArray[0, {siteNumber, siteNumber}];
Table[
        If[{a,b}!={1,1}&&{a,b}!={L1,L2},H[[getIndex[a,b,1],getIndex[a,b,2]]]=Jz];
        If[a!=1,H[[getIndex[a-1,b,2],getIndex[a,b,1]]]=Jx];
        If[b!=1,H[[getIndex[a,b-1,2],getIndex[a,b,1]]]=Jy],
        {a, L1},
        {b, L2}];
H = H + Transpose[H];
Energy = Total[Select[Sort[Eigenvalues[H]], #<0&]];
Print[Energy/siteNumber];
