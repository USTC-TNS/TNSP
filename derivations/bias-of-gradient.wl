#!/usr/bin/env wolframscript

exact=Sum[p[ss] r[ss] e[ss]d[ss],{ss,S}]/Sum[p[ss] r[ss],{ss,S}]-Sum[p[ss] r[ss] d[ss],{ss,S}]/Sum[p[ss] r[ss],{ss,S}]Sum[p[ss] r[ss] e[ss],{ss,S}]/Sum[p[ss] r[ss],{ss,S}];

hat=Sum[q[ss] r[ss] e[ss]d[ss],{ss,S}]/Sum[q[ss] r[ss],{ss,S}]-Sum[q[ss] r[ss] d[ss],{ss,S}]/Sum[q[ss] r[ss],{ss,S}]Sum[q[ss] r[ss] e[ss],{ss,S}]/Sum[q[ss] r[ss],{ss,S}];

diff=n/(n-1) hat-exact;

dd=D[diff,q[s],q[t]];

expect=(diff/.q[s_]:>p[s])+1/(2 n) Sum[Expand[(KroneckerDelta[s,t] p[s]-p[s] p[t]) dd],{s,S},{t,S}];

unwrapPattern[x_]:=x;
unwrapPattern[x_Pattern]:=x[[1]];

angleRule=Sequence@@Flatten@Table[
  Sum[Evaluate[p[ss_] r[ss_]^er e[ss_]^ee d[ss_]^ed],{ss_,S}]:>Evaluate[AngleBracket[r^unwrapPattern[er] \[CapitalEpsilon]^unwrapPattern[ee] \[CapitalDelta]^unwrapPattern[ed]]],
  {er,{0,1,2,nr_}},
  {ee,{0,1,2,nr_}},
  {ed,{0,1,2,nr_}}
];
forceNumber=Sequence@@{
  Sum[n a_,r__]:>n Sum[a,r],
  Sum[n^e_ a_,r__]:>n^e Sum[a,r],
  Sum[a_/n,r__]:>1/n Sum[a,r],
  Sum[(n-1) a_,r__]:>(n-1) Sum[a,r],
  Sum[(n-1)^e_ a_,r__]:>(n-1)^e Sum[a,r],
  Sum[ a_/(-1+n),r__]:>1/(n-1) Sum[a,r]
};
likeNumber=Sequence@@{
  Sum[i_?NumberQ a_,r__]:>i Sum[a,r],
  Sum[i_?NumberQ^n_ a_,r__]:>i^n Sum[a,r],
  Sum[i_AngleBracket a_,r__]:>i Sum[a,r],
  Sum[i_AngleBracket^n_ a_,r__]:>i^n Sum[a,r],
  forceNumber
};

Expect=Simplify[expect//.{
  q[s_]:>p[s],
  likeNumber,
  angleRule,
  Sum[a_Plus,b__]:>(Sum[#,b]&/@a),
  Sum[a_ KroneckerDelta[i_,j_],rb___,{j_,S},re___]:>If[Length[{re,rb}]!=0,Sum[a/. j->i,rb,re],a/. j->i],
  Sum[u_Times,{s_,S},{t_,S}]:>Sum[Select[u,#[[1]]==s&],{s,S}] Sum[Select[u,#[[1]]==t&],{t,S}] Error@@Select[u,#[[1]]!=t&&#[[1]]!=s&]
}];


toPsi=Sequence@@Flatten@Table[
  If[SameQ[0,ee]&&SameQ[0,ed],
     Unevaluated[Sequence[]],
     Subscript[\[LeftAngleBracket]r \[CapitalEpsilon]^ee \[CapitalDelta]^ed\[RightAngleBracket], p]:>Evaluate[Subscript[AngleBracket[\[CapitalEpsilon]^unwrapPattern[ee] \[CapitalDelta]^unwrapPattern[ed]], \[Psi]] Subscript[\[LeftAngleBracket]r\[RightAngleBracket], p]]
  ],
  {ee,{0,1,2,nr_}},
  {ed,{0,1,2,nr_}}
];
toRPsi=Sequence@@Flatten@Table[
  Subscript[\[LeftAngleBracket]r^2 \[CapitalEpsilon]^ee \[CapitalDelta]^ed\[RightAngleBracket], p]:>Evaluate[Subscript[AngleBracket[r \[CapitalEpsilon]^unwrapPattern[ee] \[CapitalDelta]^unwrapPattern[ed]], \[Psi]] Subscript[\[LeftAngleBracket]r\[RightAngleBracket], p]],
  {ee,{0,1,2,nr_}},
  {ed,{0,1,2,nr_}}
];

result1=Expand[Simplify[(n-1)Expect/.\[LeftAngleBracket]x_\[RightAngleBracket]:>Subscript[\[LeftAngleBracket]x\[RightAngleBracket], p]//.{
  toPsi,
  toRPsi,
  Subscript[\[LeftAngleBracket]r a_\[RightAngleBracket], s_]:>Subscript[\[LeftAngleBracket]r' a\[RightAngleBracket], s] Subscript[\[LeftAngleBracket]r\[RightAngleBracket], p],
  Subscript[\[LeftAngleBracket]r\[RightAngleBracket], \[Psi]]:>Subscript[\[LeftAngleBracket]r'\[RightAngleBracket], \[Psi]] Subscript[\[LeftAngleBracket]r\[RightAngleBracket], p]
}]];

G=(\[CapitalEpsilon]-Subscript[\[LeftAngleBracket]\[CapitalEpsilon]\[RightAngleBracket], \[Psi]])(\[CapitalDelta]-Subscript[\[LeftAngleBracket]\[CapitalDelta]\[RightAngleBracket], \[Psi]]);

result2=Subscript[\[LeftAngleBracket]Expand[G]\[RightAngleBracket], \[Psi]]+Subscript[\[LeftAngleBracket]r'\[RightAngleBracket], \[Psi]] Subscript[\[LeftAngleBracket]Expand[G]\[RightAngleBracket], \[Psi]]-2Subscript[\[LeftAngleBracket]Expand[r' G]\[RightAngleBracket], \[Psi]]//.{
  Subscript[\[LeftAngleBracket]u_Plus\[RightAngleBracket], \[Psi]]:>(Subscript[\[LeftAngleBracket]#\[RightAngleBracket], \[Psi]]&/@u),
  Subscript[\[LeftAngleBracket]a_  Subscript[\[LeftAngleBracket]b_ \[RightAngleBracket], \[Psi]]\[RightAngleBracket], \[Psi]]:>Subscript[\[LeftAngleBracket]a \[RightAngleBracket], \[Psi]] Subscript[\[LeftAngleBracket]b\[RightAngleBracket], \[Psi]],
  Subscript[\[LeftAngleBracket] Subscript[\[LeftAngleBracket]b_ \[RightAngleBracket], \[Psi]]\[RightAngleBracket], \[Psi]]:>Subscript[\[LeftAngleBracket]b\[RightAngleBracket], \[Psi]],
  Subscript[\[LeftAngleBracket]a_?NumberQ b_\[RightAngleBracket], \[Psi]]:>a Subscript[\[LeftAngleBracket]b\[RightAngleBracket], \[Psi]]
};

Print[result1-result2//Simplify];
