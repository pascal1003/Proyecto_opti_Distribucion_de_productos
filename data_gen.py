import random
import csv

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

# Generate N random pairs
N = 20
pairs = [(random.uniform(-15, 15), random.uniform(-15, 15)) for _ in range(N)]


aristas = []

for p in pairs:
    aristas_saliendo_de_p = []
    dist_saliendo_de_p = []
    for q in pairs:
        if p != q:
            aristas_saliendo_de_p.append((p, q))
            dist_saliendo_de_p.append(dist(p, q))
    aristas.append([p, aristas_saliendo_de_p, dist_saliendo_de_p])

# Save aristas to CSV
def save_aristas_to_csv(aristas, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['source_x', 'source_y', 'dest_x', 'dest_y', 'distance'])
        for entry in aristas:
            p = entry[0]
            aristas_saliendo_de_p = entry[1]
            dist_saliendo_de_p = entry[2]
            for idx, (p1, q) in enumerate(aristas_saliendo_de_p):
                distance = dist_saliendo_de_p[idx]
                writer.writerow([p1[0], p1[1], q[0], q[1], distance])

save_aristas_to_csv(aristas, 'aristas.csv')

# Read aristas from CSV and reconstruct
def read_aristas_from_csv(filename):
    aristas_dict = {}
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            p = (float(row['source_x']), float(row['source_y']))
            q = (float(row['dest_x']), float(row['dest_y']))
            distance = float(row['distance'])
            if p not in aristas_dict:
                aristas_dict[p] = [[], []]
            aristas_dict[p][0].append((p, q))
            aristas_dict[p][1].append(distance)
    # Convert dict to list format as original
    aristas = [[p, aristas_dict[p][0], aristas_dict[p][1]] for p in aristas_dict]
    return aristas

# Example usage:
# reconstructed_aristas = read_aristas_from_csv('aristas.csv')


reconstructed_aristas = read_aristas_from_csv('aristas.csv')

def compare_aristas(a1, a2):
    if len(a1) != len(a2):
        return False
    for entry1, entry2 in zip(a1, a2):
        if entry1[0] != entry2[0]:
            return False
        if entry1[1] != entry2[1]:
            return False
        # Compare distances with tolerance for floating point errors
        if not all(abs(d1 - d2) < 1e-6 for d1, d2 in zip(entry1[2], entry2[2])):
            return False
    return True

if compare_aristas(aristas, reconstructed_aristas):
    print("Aristas were saved and loaded correctly.")
else:
    print("Mismatch found between original and loaded aristas.")