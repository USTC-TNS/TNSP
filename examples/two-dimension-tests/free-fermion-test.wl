n = 4;
m = 4;
links = Flatten[{Table[{(i - 1)*m + j, (i - 1 + 1)*m + j}, {i, n - 1}, {j, m}],
                 Table[{(i - 1)*m + j, (i - 1)*m + (j + 1)}, {i, n}, {j, m - 1}]},
                2];
matrix = Table[If[Or[MemberQ[links, {i, j}], MemberQ[links, {j, i}]], 1., 0.], {i, n m}, {j, n m}];
Print[ExportString[Sort[Eigenvalues[matrix]],"Table"]]
res = Table[{p, Total[Sort[Eigenvalues[matrix]][[;; p]]]/(n m)}, {p, n m}];
Print[ExportString[res, "Table"]]
