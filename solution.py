# Megoldásomhoz több internetes forrást (stackoverflow), illetve a Chat-GPT 3.5 szolgáltatásait vettem igénybe
import numpy as np #(működik a Moodle-ben is)

######################## 1. feladat, entrópiaszámítás #########################

# Az entrópiaszámítás függvény, két kategória entrópiáját számolja ki
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    total = n_cat1 + n_cat2
    if total == 0:
        return 0

    p_cat1 = n_cat1 / total
    p_cat2 = n_cat2 / total

    # Ha az egyik kategória gyakorisága 0, akkor az entrópia 0
    if p_cat1 == 0 or p_cat2 == 0:
        return 0

    entropy = - (p_cat1 * np.log2(p_cat1) + p_cat2 * np.log2(p_cat2))
    return entropy

###################### 2. feladat, optimális szeparáció #######################

# A legjobb szeparáció meghatározása az információs nyereség alapján
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    best_separation_feature = 0
    best_separation_value = 0.0
    best_information_gain = 0.0

    num_features = features.shape[1]

    for feature in range(num_features):
        unique_values = np.unique(features[:, feature])

        for value in unique_values:
            left_indices = features[:, feature] <= value
            right_indices = features[:, feature] > value

            left_labels = labels[left_indices]
            right_labels = labels[right_indices]

            left_entropy = get_entropy(np.sum(left_labels == 0), np.sum(left_labels == 1))
            right_entropy = get_entropy(np.sum(right_labels == 0), np.sum(right_labels == 1))

            total_entropy = (
                    (left_labels.size / labels.size) * left_entropy
                    + (right_labels.size / labels.size) * right_entropy
            )

            information_gain = get_entropy(np.sum(labels == 0), np.sum(labels == 1)) - total_entropy

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_separation_feature = feature
                best_separation_value = value

    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
# A döntési fa építése rekurzívan
def build_decision_tree(data: np.ndarray):
    labels = data[:, -1]
    features = data[:, :-1]

    best_feature, best_value = get_best_separation(features, labels)

    decision_tree = {
        'feature_index': best_feature,
        'value': best_value,
        'left': None,
        'right': None
    }

    left_indices = features[:, best_feature] <= best_value
    right_indices = features[:, best_feature] > best_value

    left_data = data[left_indices]
    right_data = data[right_indices]

    if len(np.unique(left_data[:, -1])) == 1:
        decision_tree['left'] = int(np.unique(left_data[:, -1])[0])
    else:
        decision_tree['left'] = build_decision_tree(left_data)

    if len(np.unique(right_data[:, -1])) == 1:
        decision_tree['right'] = int(np.unique(right_data[:, -1])[0])
    else:
        decision_tree['right'] = build_decision_tree(right_data)

    return decision_tree

# A döntés fa alapján döntés meghozatala egy mintára
def make_decision(tree, sample):
    while isinstance(tree, dict):
        if sample[tree['feature_index']] <= tree['value']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree

def main():
    # Tanító adatok beolvasása
    train_data = np.loadtxt('train.csv', delimiter=',')

    # Döntési fa felépítése a tanító adatok alapján
    decision_tree = build_decision_tree(train_data)

    # Teszt adatok beolvasása és eredmények kiírása
    test_data = np.loadtxt('test.csv', delimiter=',')
    with open('results.csv', 'w') as file:
        for sample in test_data:
            decision = make_decision(decision_tree, sample)
            file.write(f"{int(decision)}\n")

if __name__ == "__main__":
    main()
