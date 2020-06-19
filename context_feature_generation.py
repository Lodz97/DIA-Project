
def generate_context_feature():
    name = ["woman", "man_eu", "man_usa"]
    feature_list = []
    for i in range(len(name)-1):
        for j in range(i+1, len(name)):
            feature_list.append([name[i], name[j]])
    l = [[element, list(set(name) - set(element))] for element in feature_list]
    l = [name] + l
    l.append([[element] for element in name])
    return l


if __name__ == "__main__":
    print(generate_context_feature())
