import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats


def generate_matrix(n, filename):
    matrix = np.random.randint(0, 10, size=(n, n))
    with open(filename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')
    return matrix


def verify_with_numpy(n):
    A = np.loadtxt("data/matrix_a.txt", skiprows=1, dtype=int)
    B = np.loadtxt("data/matrix_b.txt", skiprows=1, dtype=int)
    C = np.loadtxt("data/result_cpp.txt", skiprows=1, dtype=int)
    return np.array_equal(C, A @ B)


def run_multiplier_mpi(executable_path, num_processes, n):
    mpi_path = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

    try:
        full_exe_path = os.path.abspath(executable_path)

        cmd = [mpi_path, "-n", str(num_processes), full_exe_path]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        if result.returncode != 0:
            print(f"Ошибка MPI (Код {result.returncode}): {result.stderr}")
            return None

        if verify_with_numpy(n):
            print("  верификация пройдена")
        else:
            print("  ОШИБКА ВЕРИФИКАЦИИ!")

        for line in result.stdout.splitlines():
            if "Execution time:" in line:
                return float(line.split(':')[1].strip().split()[0])
    except FileNotFoundError:
        print(f"Критическая ошибка: Не найден mpiexec по пути {mpi_path}")
    except Exception as e:
        print(f"Ошибка: {e}")
    return None


def main():
    os.makedirs("data", exist_ok=True)
    executable = "matrix_mult_mpi.exe"

    sizes = [208, 400, 800, 1200, 1600, 2000]
    process_counts = [1, 2, 4, 8, 16]
    runs = 5
    results = {}

    for n in sizes:
        print(f"\nРазмер {n}x{n}")
        generate_matrix(n, "data/matrix_a.txt")
        generate_matrix(n, "data/matrix_b.txt")

        for proc in process_counts:
            if n % proc != 0:
                print(f"Пропуск: {n} не делится на {proc}")
                continue

            print(f"Процессов: {proc}")
            times = []
            for i in range(runs):
                t = run_multiplier_mpi(executable, proc, n)
                if t:
                    times.append(t)
                    print(f"  {t:.4f} сек")

            if times:
                results[(n, proc)] = np.mean(times)

    plt.figure(figsize=(10, 6))
    for p in process_counts:
        valid_sizes = [n for n in sizes if (n, p) in results]
        means = [results[(n, p)] for n in valid_sizes]
        if valid_sizes:
            plt.plot(valid_sizes, means, 'o-', label=f'{p} процессов')

    plt.xlabel("Размер матрицы N")
    plt.ylabel("Время выполнения (сек)")
    plt.title("Производительность матричного умножения (MPI)")
    plt.legend()
    plt.grid(True)
    plt.savefig("time_plot_mpi.png")
    plt.show()


if __name__ == "__main__":
    main()