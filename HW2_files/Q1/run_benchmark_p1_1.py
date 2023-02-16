import sysfs_paths as sysfs
import subprocess
import time


def get_avail_freqs(cluster):
    """
    Obtain the available frequency for a cpu. Return unit in khz by default!
    """
    # Read cpu freq from sysfs_paths.py
    freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]


def get_cluster_freq(cluster_num):
    """
    Read the current cluster freq. cluster_num must be 0 (little) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
        return int(f.read().strip())


def set_user_space(clusters=None):
    """
    Set the system governor as 'userspace'. This is necessary before you can change the
    cluster/cpu freq to customized values
    """
    print("Setting userspace")
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')


def set_cluster_freq(cluster_num, frequency):
    """
    Set customized freq for a cluster. Accepts frequency in khz as int or string
    """
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))


print('Available freqs for LITTLE cluster:', get_avail_freqs(0))
print('Available freqs for big cluster:', get_avail_freqs(4))
set_user_space()
set_cluster_freq(4, 2000000)   # big cluster
set_user_space()
set_cluster_freq(0, 20000)     # small cluster
# print current freq for the big cluster
print('Current freq for big cluster:', get_cluster_freq(4))

# execution of your benchmark
start=time.time()
# run the benchmark
measurement = subprocess.Popen(['python', 'measurement.py'])
command = "taskset --all-tasks 0x10 /home/student/HW2_files/TPBench.exe"   # 0x20: core 5
proc_ben = subprocess.call(command.split())
measurement.kill()
total_time = time.time() - start