n = 10;
res = Table[{p,
             Total[Sort[Eigenvalues[Table[If[Abs[i - j] == 1, 1., 0.], {i, n}, {j, n}]]][[;; p]]]/n},
            {p, n - 1}]
Print[res]
