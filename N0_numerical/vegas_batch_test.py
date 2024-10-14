import numpy as np
import vegas

def test_batch_integration():
    call_count = 0
    batch_sizes = []

    @vegas.batchintegrand
    def batch_integrand(x):
        nonlocal call_count, batch_sizes
        call_count += 1
        batch_sizes.append(x.shape[0] if x.ndim > 1 else 1)
        return np.sum(x**2, axis=1)

    integrator = vegas.Integrator([[0, 1], [0, 1]])
    result = integrator(batch_integrand, nitn=10, neval=10000)

    print(f"Result: {result}")
    print(f"Total calls: {call_count}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Max batch size: {max(batch_sizes)}")
    print(f"Min batch size: {min(batch_sizes)}")
    print(f"Average batch size: {sum(batch_sizes) / len(batch_sizes):.2f}")

if __name__ == "__main__":
    test_batch_integration()