def mean(obj):
    return sum(obj)/len(obj)

if __name__ == "__main__":
    J_dict, F_dict = {}, {}
    with open("RVOS.txt", "r") as f:
        for line in f.readlines():
            line = line.strip("\n").split(",")
            try:
                assert len(line) > 1
            except:
                print(line)
                continue
            key = line[0]
            J, F = float(line[-2]), float(line[-1])
            if key not in J_dict:
                J_dict[key] = []
            if key not in F_dict:
                F_dict[key] = []
            J_dict[key].append(J)
            F_dict[key].append(F)
    JF_dict = {}
    for key in J_dict.keys():
        J = mean(J_dict[key])
        F = mean(F_dict[key])
        JF_dict[key] = (J+F)/2
    print(sorted(JF_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:10])  