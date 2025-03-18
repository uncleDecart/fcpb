import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import time
import numpy as np
import pandas as pd


# Training function
def train_model(model, trainset, batch_size, epochs=1):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_forward = 0
    total_backward = 0
    timing = {"forward": [], "backward": []}

    if torch.cuda.is_available():
        for epoch in range(epochs):
            total_forward = 0
            total_backward = 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                start_forward = torch.cuda.Event(enable_timing=True)
                end_forward = torch.cuda.Event(enable_timing=True)
                start_backward = torch.cuda.Event(enable_timing=True)
                end_backward = torch.cuda.Event(enable_timing=True)

                start_forward.record()
                outputs = model(images)
                end_forward.record()
                torch.cuda.synchronize()
                forward_time = start_forward.elapsed_time(end_forward)

                start_backward.record()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                end_backward.record()
                torch.cuda.synchronize()
                backward_time = start_backward.elapsed_time(end_backward)

                total_forward += forward_time
                total_backward += backward_time

            timing["forward"].append(total_forward / (len(trainloader)))
            timing["backward"].append(total_backward / (len(trainloader)))

    else:
        for epoch in range(epochs):
            total_forward = 0
            total_backward = 0
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                start_forward = time.time()
                outputs = model(images)
                forward_time = time.time() - start_forward

                start_backward = time.time()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                backward_time = time.time() - start_backward

                total_forward += forward_time
                total_backward += backward_time

            timing["forward"].append(total_forward / (len(trainloader)))
            timing["backward"].append(total_backward / (len(trainloader)))

    return model, timing


# Throughput Comparison
def benchmark(model, testset, batch_size, num_batches=10):
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_time = 0
    # branching only once not relying on branch predictor
    if torch.cuda.is_available():
        with torch.no_grad():
            for i, (images, _) in enumerate(testloader):
                if i >= num_batches:
                    break
                images = images.to(device)

                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()
                _ = model(images)
                end_time.record()
                torch.cuda.synchronize()
                total_time += start_time.elapsed_time(end_time)
    else:
        with torch.no_grad():
            for i, (images, _) in enumerate(testloader):
                if i >= num_batches:
                    break
                images = images.to(device)

                start_time = time.time()
                _ = model(images)
                end_time = time.time()
                total_time += (
                    end_time - start_time
                ) * 1000  # Convert seconds to milliseconds

    return total_time / num_batches


# Returns dict in following format
# naive_times - list average times to run inference on naive model for "num_trials" times
# optimized_times - list of average times to run inference on optimized model for "num_epochs" times
# optimized_tt_fwd - list of average times to run forward propagation on training model for "num_epochs" times
# optimized_tt_bckwd - list of average times to run backward propagation on training model for "num_epochs"
# use _mean and _std postfix to get mean and standard deviation for lists above, e.g. navie_times_mean, naive_times_std
def compare_performance(
    naive,
    optimized,
    trainset,
    testset,
    tile_sizes=[32, 64, 128],
    num_trials=16,
    num_epochs=16,
):
    results = {}
    for tile_size in tile_sizes:
        print(f"\nBatch Size: {tile_size}")
        naive_model, naive_tt = train_model(
            naive,
            trainset,
            batch_size=tile_size,
            epochs=num_epochs,
        )
        optimized_model, optimized_tt = train_model(
            optimized,
            trainset,
            batch_size=tile_size,
            epochs=num_epochs,
        )
        optimized_model = torch.jit.script(
            optimized_model
        )  # Apply JIT for further optimization

        naive_times = []
        optimized_times = []

        for _ in range(num_trials):
            naive_times.append(benchmark(naive_model, testset, batch_size=tile_size))
            optimized_times.append(
                benchmark(optimized_model, testset, batch_size=tile_size)
            )

        naive_mean, naive_std = np.mean(naive_times), np.std(naive_times)
        optimized_mean, optimized_std = np.mean(optimized_times), np.std(
            optimized_times
        )

        naive_tt_mean_fwd, naive_tt_std_fwd = np.mean(naive_tt["forward"]), np.std(
            naive_tt["forward"]
        )
        optimized_tt_mean_fwd, optimized_tt_std_fwd = np.mean(
            optimized_tt["forward"]
        ), np.std(optimized_tt["forward"])

        naive_tt_mean_bckwd, naive_tt_std_bckwd = np.mean(naive_tt["backward"]), np.std(
            naive_tt["backward"]
        )
        optimized_tt_mean_bckwd, optimized_tt_std_bckwd = np.mean(
            optimized_tt["backward"]
        ), np.std(optimized_tt["backward"])

        results[tile_size] = {
            "naive_times": naive_times,
            "naive_mean": naive_mean,
            "naive_std": naive_std,
            "optimized_times": optimized_times,
            "optimized_mean": optimized_mean,
            "optimized_std": optimized_std,
            "naive_tt_fwd": naive_tt["forward"],
            "naive_tt_fwd_mean": naive_tt_mean_fwd,
            "naive_tt_fwd_std": naive_tt_std_fwd,
            "optimized_tt_fwd": optimized_tt["forward"],
            "optimized_tt_fwd_mean": optimized_tt_mean_fwd,
            "optimized_tt_fwd_std": optimized_tt_std_fwd,
            "naive_tt_bckwd": naive_tt["backward"],
            "naive_tt_bckwd_mean": naive_tt_mean_bckwd,
            "naive_tt_bckwd_std": naive_tt_std_bckwd,
            "optimized_tt_bckwd": optimized_tt["backward"],
            "optimized_tt_bckwd_mean": optimized_tt_mean_bckwd,
            "optimized_tt_bckwd_std": optimized_tt_std_bckwd,
        }

    return results
