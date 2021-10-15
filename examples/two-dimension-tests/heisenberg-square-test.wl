n = 4;
m = 3;
generate[i_, j_, n_] := Sum[KroneckerProduct @@ Table[If[And[x != i, x != j], IdentityMatrix[2], PauliMatrix[d]/2.], {x, n}], {d, 3}];
H = Total[Flatten[{Table[generate[(i - 1)*m + j, (i - 1 + 1)*m + j, n m], {i, n - 1}, {j, m}], Table[generate[(i - 1)*m + j, (i - 1)*m + (j + 1), n m], {i, n}, {j, m - 1}]}, 2]];
e = Min[Eigenvalues[H]]/(n m)
Print[e]
