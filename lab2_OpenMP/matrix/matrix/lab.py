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


def run_multiplier(executable_path, num_threads):
    """Запускает программу умножения с заданным числом потоков и возвращает время."""
    try:
        output = subprocess.check_output([executable_path, str(num_threads)], stderr=subprocess.STDOUT, text=True)
        for line in output.splitlines():
            if "Execution time:" in line:
                return float(line.split(':')[1].strip().split()[0])
    except Exception as e:
        print(f"Ошибка выполнения: {e}")
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
            return True
        else:
            return False
    except:
        return False


def main():
    os.makedirs("data", exist_ok=True)

    executable = "matrix_mult.exe" if os.name == 'nt' else "./matrix_mult"
    if not os.path.exists(executable):
        print("Ошибка: файл", executable, "не найден")
        sys.exit(1)

    sizes = [200, 400, 800, 1200, 1600, 2000]
    threads_list = [1, 2, 4, 8, 16]
    runs = 5

    results = {}

    for n in sizes:
        print(f"\nРазмер {n}x{n}")

        A = generate_matrix(n, "data/matrix_a.txt")
        B = generate_matrix(n, "data/matrix_b.txt")

        for threads in threads_list:
            print(f"\nПотоков: {threads}")
            times = []
            verified = False

            for i in range(runs):
                print(f"  Запуск {i + 1}/{runs}: ", end="", flush=True)
                t = run_multiplier(executable, threads)
                if t is not None:
                    times.append(t)
                    print(f"{t:.4f} сек")
                else:
                    print("ошибка")

            if times:
                results[(n, threads)] = times
                if os.path.exists("data/result_cpp.txt") and not verified:
                    if verify_result(A, B, "data/result_cpp.txt"):
                        print("  верификация пройдена")
                        verified = True
                    else:
                        print("  ошибка верификации")

    print("\nРЕЗУЛЬТАТЫ")

    plt.figure(figsize=(10, 6))

    for threads in threads_list:
        means = []
        errors = []
        valid_sizes = []

        print(f"\n--- Для {threads} потоков ---")
        for n in sizes:
            if (n, threads) in results:
                times = results[(n, threads)]
                mean = np.mean(times)
                std = np.std(times, ddof=1)

                if len(times) > 1 and std > 0:
                    t_crit = stats.t.ppf(0.975, len(times) - 1)
                    ci = t_crit * std / np.sqrt(len(times))
                else:
                    ci = 0

                means.append(mean)
                errors.append(ci)
                valid_sizes.append(n)

                print(f"Размер {n}: {mean:.4f} сек (±{ci:.4f})")

        if valid_sizes:
            plt.errorbar(valid_sizes, means, yerr=errors, fmt='o-', capsize=5, capthick=2, label=f'{threads} потоков')

    plt.xlabel("Размер матрицы N")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Зависимость времени умножения от размера матрицы и потоков OpenMP")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_plot_omp.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()