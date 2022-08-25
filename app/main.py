import sys
from io import StringIO
import contextlib
import subprocess

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

if __name__ == "__main__":
    with stdoutIO() as out1:
        exec(open('app/expert_fscore.py').read())

    with open("app/output2.txt", "w+") as output2:
        subprocess.call(["python", "./app/ml_predictions.py"], stdout=output2)
    with open("app/output2.txt") as f:
        lines2 = f.readlines()

    with open("app/output3.txt", "w+") as output3:
        subprocess.call(["python", "./app/output_parser.py", "./app/output2.txt"], stdout=output3)
    with open("app/output3.txt") as f:
        lines3 = f.readlines()

    print("========================================================")
    print("Expert Performance Measures")
    print("========================================================")
    print("%s" % (out1.getvalue()))

    print("========================================================")
    print("ML Models Performance Measures")
    print("========================================================")
    print("%s" % ("\n".join(lines2)))

    print("========================================================")
    print("Formatted ML Models Performance Measures")
    print("========================================================")
    print("%s" % ("\n".join(lines3)))