import matplotlib.pyplot as plt

def parse_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    results = {}
    for line in lines:
        if 'Time' in line:
            key, value = line.split(': ')
            results[key] = float(value.split()[0])
    return results

def display_results(results):
    tasks = results.keys()
    times = results.values()

    plt.bar(tasks, times)
    plt.xlabel('Benchmark Tasks')
    plt.ylabel('Time (seconds)')
    plt.title('GPU Benchmark Results')
    plt.show()

if __name__ == "__main__":
    results = parse_results('results.txt')
    display_results(results)