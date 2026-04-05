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


def run_multiplier(executable_path):
    """Запускает программу умножения и возвращает время выполнения."""
    try:
        output = subprocess.check_output([executable_path], stderr=subprocess.STDOUT, text=True)
        for line in output.splitlines():
            if "Execution time:" in line:
                return float(line.split(':')[1].strip().split()[0])
    except:
        return None
    return None


def verify_result(a_matrix, b_matrix, result_file):
    """Сравнивает результат C++ с эталонным от NumPy."""
    try:
        with open(result_file, 'r') as f:
            n = int(f.readline())
            cpp_matrix = []
            for _ in range(n):
                cpp_matrix.append(list(map(int, f.readline().split())))
        cpp_matrix = np.array(cpp_matrix)
        expected = np.dot(a_matrix, b_matrix)

        if np.array_equal(cpp_matrix, expected):
            print("Верификация: успешно")
            return True
        else:
            print("Верификация: не совпадает")
            return False
    except:
        return False


def main():
    os.makedirs("data", exist_ok=True)

    executable = "matrix_mult.exe" if os.name == 'nt' else "./matrix_mult"
    if not os.path.exists(executable):
        print("Ошибка: файл", executable, "не найден")
        sys.exit(1)

    sizes = [100, 200, 300, 400, 500, 700, 1000]
    runs = 10

    results = {}

    for n in sizes:
        print(f"\nРазмер {n}x{n}")

        A = generate_matrix(n, "data/matrix_a.txt")
        B = generate_matrix(n, "data/matrix_b.txt")

        times = []

        for i in range(runs):
            print(f"  Запуск {i + 1}/{runs}: ", end="", flush=True)
            t = run_multiplier(executable)
            if t:
                times.append(t)
                print(f"{t:.4f} сек")
            else:
                print("ошибка")

        if times:
            results[n] = times
            if os.path.exists("data/result_cpp.txt"):
                verify_result(A, B, "data/result_cpp.txt")

    print("РЕЗУЛЬТАТЫ")

    means = []
    errors = []

    for n in sizes:
        if n in results:
            times = results[n]
            mean = np.mean(times)
            std = np.std(times, ddof=1)

            t_crit = stats.t.ppf(0.975, len(times) - 1)
            ci = t_crit * std / np.sqrt(len(times))

            means.append(mean)
            errors.append(ci)

            print(f"\nРазмер {n}:")
            print(f"  Среднее: {mean:.4f} сек")
            print(f"  Доверительный интервал (95%): [{mean - ci:.4f}, {mean + ci:.4f}] сек")

    if means:
        plt.figure()
        plt.errorbar([n for n in sizes if n in results], means, yerr=errors,
                     fmt='o-', capsize=5, capthick=2)
        plt.xlabel("Размер матрицы N")
        plt.ylabel("Время выполнения (сек)")
        plt.title("Зависимость времени умножения от размера матрицы")
        plt.grid(True)
        plt.savefig("time_plot.png")
        plt.show()

if __name__ == "__main__":
    main()
