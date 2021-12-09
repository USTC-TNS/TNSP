import TAT
import tetragono as tet
import pickle


def load(name):
    with open(name, "rb") as file:
        return pickle.load(file)


def load_with_step(sj, me):
    return load(f"SJ{sj}ME{me}")


s = {}
for sj in [1, 2, 4, 8]:
    for me in range(1, 10):
        s[sj, me] = load_with_step(sj, me)

for total in range(10):
    should_same = []
    for [sj, me], state in s.items():
        if sj + me == total:
            should_same.append((sj, me, state))
    if len(should_same) > 1:
        sja, mea, a = should_same.pop()
        for sjb, meb, b in should_same:
            print(sja, mea, sjb, meb)
            # a.initialize_auxiliaries(-1)
            # b.initialize_auxiliaries(-1)
            # print("ED", a.observe_energy() - b.observe_energy())
            for l1 in range(a.L1):
                for l2 in range(a.L2):
                    if l1 != 0:
                        ea = a.environment[l1, l2, "U"]
                        eb = b.environment[l1, l2, "U"]
                        try:
                            print("SD", (ea - eb.transpose(ea.names)).norm_max())
                        except:
                            print("SD", "EDGE PROBLEM")
                            #exit()
                    if l2 != 0:
                        ea = a.environment[l1, l2, "L"]
                        eb = b.environment[l1, l2, "L"]
                        try:
                            print("SD", (ea - eb.transpose(ea.names)).norm_max())
                        except:
                            print("SD", "EDGE PROBLEM")
                            #exit()
