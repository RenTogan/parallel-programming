import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

def generate_matrix(n, filename):
    """Генерирует случайную целочисленную матрицу n x n и сохраняет в файл."""
    matrix = np.random.randint(0, 10, size=(n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')
    return matrix

def run_multiplier(executable, block_size):
    """Запускает CUDA-программу с заданным размером блока и возвращает время."""
    try:
        output = subprocess.check_output([executable, str(block_size)], stderr=subprocess.STDOUT, text=True)
        for line in output.splitlines():
            if "Execution time:" in line:
                return float(line.split(':')[1].strip().split()[0])
    except Exception as e:
        print(f"Ошибка выполнения: {e}")
        return None
    return None

def verify_result(a_matrix, b_matrix, result_file):
    """Сравнивает результат C++ с эталоном от NumPy."""
    try:
        with open(result_file, 'r') as f:
            n = int(f.readline())
            cpp_matrix = []
            for _ in range(n):
                cpp_matrix.append(list(map(int, f.readline().split())))
        cpp_matrix = np.array(cpp_matrix)
        expected = np.dot(a_matrix, b_matrix)
        return np.array_equal(cpp_matrix, expected)
    except:
        return False

def main():
    os.makedirs("data", exist_ok=True)

    executable = "matrix_mult_cuda.exe"
    if not os.path.exists(executable):
        print("Ошибка: файл", executable, "не найден")
        sys.exit(1)

    sizes = [200, 400, 800, 1200, 1600, 2000]
    block_sizes = [8, 16, 32]
    runs = 5

    results = {}

    for n in sizes:
        print(f"\n=== Размер матрицы {n}x{n} ===")
        A = generate_matrix(n, "data/matrix_a.txt")
        B = generate_matrix(n, "data/matrix_b.txt")

        for bs in block_sizes:
            print(f"  Блок {bs}x{bs}")
            times = []
            verified = False

            for run_idx in range(runs):
                print(f"    Запуск {run_idx+1}/{runs}: ", end="", flush=True)
                t = run_multiplier(executable, bs)
                if t is not None:
                    times.append(t)
                    print(f"{t:.4f} сек")
                else:
                    print("ошибка")

            if times:
                results[(n, bs)] = times
                if not verified and os.path.exists("data/result_cpp.txt"):
                    if verify_result(A, B, "data/result_cpp.txt"):
                        print("    Верификация пройдена")
                        verified = True
                    else:
                        print("    Ошибка верификации!")

    plt.figure(figsize=(10, 6))
    for bs in block_sizes:
        means = []
        errors = []
        valid_sizes = []
        for n in sizes:
            key = (n, bs)
            if key in results:
                times = results[key]
                mean = np.mean(times)
                std = np.std(times, ddof=1) if len(times) > 1 else 0.0
                if len(times) > 1 and std > 0:
                    t_crit = stats.t.ppf(0.975, len(times)-1)
                    ci = t_crit * std / np.sqrt(len(times))
                else:
                    ci = 0.0
                means.append(mean)
                errors.append(ci)
                valid_sizes.append(n)
        if valid_sizes:
            plt.errorbar(valid_sizes, means, yerr=errors, fmt='o-', capsize=5,
                         capthick=2, label=f'Блок {bs}x{bs}')

    plt.xlabel("Размер матрицы N")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Умножение матриц на GPU (CUDA) с разными размерами блоков")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_plot_cuda.png", dpi=300)
    plt.show()

    print("\n=== Сводная таблица времени (сек) ===")
    print("Размер\\Блок", end="")
    for bs in block_sizes:
        print(f"{bs:8}", end="")
    print()
    for n in sizes:
        print(f"{n:6}", end="")
        for bs in block_sizes:
            key = (n, bs)
            if key in results:
                mean = np.mean(results[key])
                print(f"{mean:8.4f}", end="")
            else:
                print("   ---   ", end="")
        print()

if __name__ == "__main__":
    main()