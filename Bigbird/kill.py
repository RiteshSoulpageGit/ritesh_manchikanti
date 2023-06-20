
import subprocess
import os
# Run the nvidia-smi command and capture the output
output = subprocess.check_output(['nvidia-smi', '-q', '-x'])

# Parse the XML output to extract the PIDs
import xml.etree.ElementTree as ET
root = ET.fromstring(output)

pids = []
for gpu in root.findall('.//gpu'):
    processes = gpu.find('processes')
    if processes is not None:
        for process in processes.findall('process_info'):
            pid = process.find('pid').text
            pids.append(int(pid))

# Print the list of PIDs
print(pids)
for i in pids:
    os.system(f'kill -9 {i}')
# import os
# pid = 1651403  # Replace with the actual process ID
# # Kill the process using pkill
# os.system(f'kill -9 {pid}')
