import subprocess


def run_cmds(_pci_id_cmd, silent=True):
    sp = subprocess.Popen(_pci_id_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    rc=sp.wait()
    out,err=sp.communicate()
    return str(out).splitlines()
